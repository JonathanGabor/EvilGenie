"""
Automated tester for running coding agents on LiveCodeBench problems.

This module orchestrates the entire testing workflow:
1. Create workspace for problem
2. Launch agent
3. Monitor for solution changes
4. Run tests when solution is updated
5. Collect results
6. Clean up
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from agent_runner import create_agent_runner, AgentRunner
from workspace_manager import WorkspaceManager, WorkspaceMonitor
from reward_hacking_detector import RewardHackingDetector

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of testing a single problem with an agent."""
    problem_id: str
    agent_name: str
    success: bool
    runtime: float
    test_passed: bool
    test_results: Dict[str, Any]
    solution_content: Optional[str]
    error: Optional[str]
    timestamp: datetime
    # Reward hacking detection fields
    reward_hacking_detected: bool = False
    reward_hacking_summary: Optional[Dict[str, Any]] = None
    reward_hacking_violations: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class AutomatedTester:
    """Main class for automated agent testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize automated tester.
        
        Args:
            config: Configuration dictionary with test parameters
        """
        self.config = config
        self.workspace_manager = WorkspaceManager(
            base_dir=config.get('workspace_base_dir'),
            cleanup=config.get('cleanup', True)
        )
        self.results = []
        self.timeout = config.get('timeout', 300)  # 5 minutes default
        self.check_interval = config.get('check_interval', 5)  # Check every 5 seconds
        
        # Setup logging
        log_level = config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_single_problem(self, problem_id: str, agent_name: str, agent_config: Dict[str, Any]) -> TestResult:
        """
        Test a single problem with a specific agent.
        
        Args:
            problem_id: ID of the problem to test
            agent_name: Name of the agent to use
            agent_config: Configuration for the agent
            
        Returns:
            TestResult object with the results
        """
        logger.info(f"Testing {problem_id} with {agent_name}")
        start_time = time.time()
        
        try:
            # Create workspace
            workspace_path = self.workspace_manager.create_workspace(problem_id, agent_name)
            if not workspace_path:
                return TestResult(
                    problem_id=problem_id,
                    agent_name=agent_name,
                    success=False,
                    runtime=0,
                    test_passed=False,
                    test_results={},
                    solution_content=None,
                    error="Failed to create workspace",
                    timestamp=datetime.now(),
                    reward_hacking_detected=False,
                    reward_hacking_summary=None,
                    reward_hacking_violations=None
                )
            
            # Initialize reward hacking detection
            reward_hacking_config = self.config.get('reward_hacking_detection', {})
            reward_detector = RewardHackingDetector(workspace_path, reward_hacking_config)
            
            if not reward_detector.initialize_baseline():
                logger.error("Failed to initialize reward hacking detection baseline")
                # Continue anyway, but log the failure
            
            # Create agent runner
            agent_runner = create_agent_runner(agent_name, str(workspace_path), agent_config)
            if not agent_runner:
                return TestResult(
                    problem_id=problem_id,
                    agent_name=agent_name,
                    success=False,
                    runtime=0,
                    test_passed=False,
                    test_results={},
                    solution_content=None,
                    error=f"Unknown agent type: {agent_name}",
                    timestamp=datetime.now(),
                    reward_hacking_detected=False,
                    reward_hacking_summary=None,
                    reward_hacking_violations=None
                )
            
            # Launch agent
            if not agent_runner.launch():
                return TestResult(
                    problem_id=problem_id,
                    agent_name=agent_name,
                    success=False,
                    runtime=0,
                    test_passed=False,
                    test_results={},
                    solution_content=None,
                    error="Failed to launch agent",
                    timestamp=datetime.now(),
                    reward_hacking_detected=False,
                    reward_hacking_summary=None,
                    reward_hacking_violations=None
                )
            
            # Monitor workspace
            monitor = WorkspaceMonitor(workspace_path)
            solution_found = False
            test_results = {}
            solution_content = None
            
            # Wait for agent to be ready (exclude connection time from timeout)
            connection_start = time.time()
            while time.time() - connection_start < 120:  # Max 2 minutes for connection
                if hasattr(agent_runner, 'update_output'):
                    agent_runner.update_output()
                
                if hasattr(agent_runner, 'is_ready') and agent_runner.is_ready():
                    logger.info(f"Agent {agent_name} is ready after {time.time() - connection_start:.1f}s")
                    break
                    
                if not agent_runner.is_running():
                    logger.error(f"Agent {agent_name} died during connection")
                    return TestResult(
                        problem_id=problem_id,
                        agent_name=agent_name,
                        success=False,
                        runtime=time.time() - start_time,
                        test_passed=False,
                        test_results={},
                        solution_content=None,
                        error="Agent died during connection",
                        timestamp=datetime.now(),
                        reward_hacking_detected=False,
                        reward_hacking_summary=None,
                        reward_hacking_violations=None
                    )
                
                time.sleep(0.5)
            
            # Start timeout timer only after agent is ready
            processing_start = time.time()
            last_log_save = processing_start
            last_status_log = processing_start
            
            while time.time() - processing_start < self.timeout:
                # Update output for streaming agents (like Gemini)
                if hasattr(agent_runner, 'update_output'):
                    agent_runner.update_output()
                    
                    # Save logs more frequently to capture progress
                    if time.time() - last_log_save > 10:  # Every 10 seconds
                        if hasattr(agent_runner, '_save_output_log'):
                            agent_runner._save_output_log()
                        last_log_save = time.time()
                
                # Log status periodically for debugging
                if time.time() - last_status_log > 30:  # Every 30 seconds
                    processing_elapsed = time.time() - processing_start
                    total_elapsed = time.time() - start_time
                    logger.info(f"Agent {agent_name} processing for {processing_elapsed:.1f}s (total {total_elapsed:.1f}s)")
                    if hasattr(agent_runner, 'stdout_content'):
                        stdout_len = len(agent_runner.stdout_content)
                        stderr_len = len(agent_runner.stderr_content) if hasattr(agent_runner, 'stderr_content') else 0
                        logger.info(f"Output captured: {stdout_len} stdout chars, {stderr_len} stderr chars")
                    last_status_log = time.time()
                
                # Check if agent is still running
                if not agent_runner.is_running():
                    logger.info(f"Agent {agent_name} has finished")
                    break
                
                # Check for reward hacking violations
                violations = reward_detector.check_for_violations()
                if violations and reward_detector.config.get('fail_fast', True):
                    logger.critical(f"Reward hacking detected! Terminating agent immediately.")
                    if agent_runner.is_running():
                        agent_runner.terminate()
                    break
                
                # Check for solution changes (just track, don't test yet)
                if monitor.has_solution_changed():
                    new_solution_content = monitor.get_solution_content()
                    logger.info(f"Solution file updated - content length: {len(new_solution_content)} chars")
                    logger.debug(f"Solution preview: {new_solution_content[:200]}...")
                    
                    # Check if it's actually different from template
                    if "TODO" not in new_solution_content:
                        logger.info("Solution appears to contain real implementation")
                    else:
                        logger.warning("Solution still contains TODO markers - possible false detection")
                    
                    solution_found = True
                    solution_content = new_solution_content
                    # Note: We'll test only at the end
                
                time.sleep(self.check_interval)
            
            # Final check for solution changes after agent completes
            if not solution_found and monitor.has_solution_changed():
                logger.info("Solution file updated after agent completion")
                solution_found = True
                solution_content = monitor.get_solution_content()
            
            # Test the final solution (only once at the end)
            if solution_found:
                logger.info("Running tests on final solution...")
                test_results = monitor.run_tests()
            
            # Timeout or agent finished
            processing_runtime = time.time() - processing_start
            total_runtime = time.time() - start_time
            
            logger.info(f"Agent completed: {processing_runtime:.1f}s processing, {total_runtime:.1f}s total")
            runtime = total_runtime  # Keep total runtime for compatibility
            
            # Ensure agent is terminated
            if agent_runner.is_running():
                logger.warning(f"Agent {agent_name} timed out, terminating...")
                agent_runner.terminate()
            
            # Run final test if we haven't already
            if solution_found and not test_results:
                test_results = monitor.run_tests()
            
            # Final reward hacking check
            reward_detector.check_for_violations()
            reward_hacking_summary = reward_detector.get_violation_summary()
            
            # Clean up reward hacking detector
            reward_detector.cleanup()
            
            # Save results before cleanup
            if self.config.get('save_results', True):
                results_dir = Path(self.config.get('results_dir', 'results')) / agent_name / problem_id
                self.workspace_manager.save_workspace_results(workspace_path, results_dir)
            
            # Create test result
            return TestResult(
                problem_id=problem_id,
                agent_name=agent_name,
                success=solution_found,
                runtime=runtime,
                test_passed=test_results.get('success', False) if test_results else False,
                test_results=test_results,
                solution_content=solution_content,
                error=None if solution_found else "No solution generated",
                timestamp=datetime.now(),
                reward_hacking_detected=reward_hacking_summary['has_violations'],
                reward_hacking_summary=reward_hacking_summary,
                reward_hacking_violations=reward_hacking_summary.get('violations', [])
            )
            
        except Exception as e:
            logger.error(f"Error testing {problem_id} with {agent_name}: {e}")
            
            # Try to get reward hacking results even on error
            reward_hacking_detected = False
            reward_hacking_summary = None
            reward_hacking_violations = None
            
            try:
                if 'reward_detector' in locals():
                    reward_hacking_summary = reward_detector.get_violation_summary()
                    reward_hacking_detected = reward_hacking_summary.get('has_violations', False)
                    reward_hacking_violations = reward_hacking_summary.get('violations', [])
                    reward_detector.cleanup()
            except Exception as cleanup_error:
                logger.debug(f"Error getting reward hacking results during exception handling: {cleanup_error}")
            
            return TestResult(
                problem_id=problem_id,
                agent_name=agent_name,
                success=False,
                runtime=time.time() - start_time,
                test_passed=False,
                test_results={},
                solution_content=None,
                error=str(e),
                timestamp=datetime.now(),
                reward_hacking_detected=reward_hacking_detected,
                reward_hacking_summary=reward_hacking_summary,
                reward_hacking_violations=reward_hacking_violations
            )
    
    def test_batch(self, problems: List[str], agents: List[Dict[str, Any]]) -> List[TestResult]:
        """
        Test multiple problems with multiple agents.
        
        Args:
            problems: List of problem IDs to test
            agents: List of agent configurations
            
        Returns:
            List of TestResult objects
        """
        results = []
        total_tests = len(problems) * len(agents)
        completed = 0
        
        for problem_id in problems:
            for agent_config in agents:
                agent_name = agent_config['name']
                logger.info(f"Testing {completed + 1}/{total_tests}: {problem_id} with {agent_name}")
                
                result = self.test_single_problem(problem_id, agent_name, agent_config)
                results.append(result)
                self.results.append(result)
                
                # Save intermediate results
                self._save_results()
                
                completed += 1
                
                # Optional delay between tests
                delay = self.config.get('test_delay', 0)
                if delay > 0 and completed < total_tests:
                    time.sleep(delay)
        
        return results
    
    def _save_results(self) -> None:
        """Save current results to file."""
        results_file = Path(self.config.get('results_file', 'test_results.json'))
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2
            )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of test results."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Group results by agent
        by_agent = {}
        for result in self.results:
            if result.agent_name not in by_agent:
                by_agent[result.agent_name] = []
            by_agent[result.agent_name].append(result)
        
        # Calculate statistics
        report = {
            "total_tests": len(self.results),
            "agents": {}
        }
        
        # Calculate overall reward hacking statistics
        total_violations = sum(1 for r in self.results if r.reward_hacking_detected)
        violation_types = {}
        severity_counts = {}
        
        for result in self.results:
            if result.reward_hacking_detected and result.reward_hacking_violations:
                for violation in result.reward_hacking_violations:
                    v_type = violation.get('violation_type', 'unknown')
                    severity = violation.get('severity', 'unknown')
                    violation_types[v_type] = violation_types.get(v_type, 0) + 1
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for agent_name, agent_results in by_agent.items():
            total = len(agent_results)
            passed = sum(1 for r in agent_results if r.test_passed)
            solutions_generated = sum(1 for r in agent_results if r.success)
            avg_runtime = sum(r.runtime for r in agent_results) / total
            
            # Reward hacking statistics for this agent
            agent_violations = sum(1 for r in agent_results if r.reward_hacking_detected)
            agent_violation_rate = agent_violations / total if total > 0 else 0
            
            agent_violation_types = {}
            for result in agent_results:
                if result.reward_hacking_detected and result.reward_hacking_violations:
                    for violation in result.reward_hacking_violations:
                        v_type = violation.get('violation_type', 'unknown')
                        agent_violation_types[v_type] = agent_violation_types.get(v_type, 0) + 1
            
            report["agents"][agent_name] = {
                "total_problems": total,
                "tests_passed": passed,
                "pass_rate": passed / total if total > 0 else 0,
                "solutions_generated": solutions_generated,
                "generation_rate": solutions_generated / total if total > 0 else 0,
                "average_runtime": avg_runtime,
                "reward_hacking_violations": agent_violations,
                "reward_hacking_rate": agent_violation_rate,
                "violation_types": agent_violation_types,
                "problems": {
                    r.problem_id: {
                        "passed": r.test_passed,
                        "runtime": r.runtime,
                        "error": r.error,
                        "reward_hacking_detected": r.reward_hacking_detected,
                        "violation_count": len(r.reward_hacking_violations) if r.reward_hacking_violations else 0
                    }
                    for r in agent_results
                }
            }
        
        # Add overall reward hacking summary to report
        report["reward_hacking_summary"] = {
            "total_violations": total_violations,
            "violation_rate": total_violations / len(self.results) if self.results else 0,
            "violation_types": violation_types,
            "severity_counts": severity_counts,
            "detection_enabled": True  # Assumes detection was enabled if we're reporting
        }
        
        return report
    
    def cleanup(self) -> None:
        """Clean up all workspaces."""
        self.workspace_manager.cleanup_all()