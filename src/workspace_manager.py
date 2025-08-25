"""
Workspace manager for creating and managing isolated problem environments.

This module handles:
- Creating temporary workspaces for each problem
- Setting up problem files using problem_setup.py
- Cleaning up workspaces after testing
- Managing multiple concurrent workspaces
"""

import os
import shutil
import tempfile
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

# Import problem setup functions
from problem_setup import setup_problem_by_id
from env_utils import build_subprocess_env

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages isolated workspaces for testing problems."""
    
    def __init__(self, base_dir: Optional[str] = None, cleanup: bool = True, release_version: str = "v6"):
        """
        Initialize workspace manager.
        
        Args:
            base_dir: Base directory for workspaces (uses temp dir if None)
            cleanup: Whether to cleanup workspaces after use
            release_version: Dataset release version to use (default: v6)
        """
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = Path(tempfile.gettempdir()) / "lcb_agent_testing"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
        self.cleanup = cleanup
        self.release_version = release_version
        self.active_workspaces = {}
        logger.debug(f"Workspace manager initialized with base dir: {self.base_dir}, release: {self.release_version}")
    
    def create_workspace(self, problem_id: str, agent_name: str, holdout_config: dict = None) -> Optional[Path]:
        """
        Create a new workspace for a problem.
        
        Args:
            problem_id: ID of the problem to test
            agent_name: Name of the agent that will use this workspace
            
        Returns:
            Path to the created workspace or None if failed
        """
        try:
            # Create unique workspace name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_name = f"{agent_name}_{problem_id}_{timestamp}"
            workspace_path = self.base_dir / workspace_name
            
            # Use direct function call instead of subprocess
            logger.debug(f"Setting up problem {problem_id} in {workspace_path}")
            result = setup_problem_by_id(
                problem_id=problem_id,
                output_dir=str(workspace_path.absolute()),
                release_version=self.release_version,  # Use configured release version directly
                verbose=True,
                holdout_config=holdout_config
            )
            
            if result is None:
                logger.error(f"Failed to setup problem workspace for {problem_id}")
                return None
            
            problem_dir, files = result
            actual_workspace = Path(problem_dir)
            
            # Verify the required files exist
            required_files = ["problem.md", "solution.py", "test.py", "test_cases.json"]
            for file_name in required_files:
                if not (actual_workspace / file_name).exists():
                    logger.error(f"Required file {file_name} not found in {actual_workspace}")
                    return None
            
            # Store workspace info
            self.active_workspaces[actual_workspace] = {
                "problem_id": problem_id,
                "agent_name": agent_name,
                "created_at": datetime.now(),
                "status": "active"
            }
            
            logger.debug(f"Created workspace: {actual_workspace}")
            return actual_workspace
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            return None
    
    def create_base_workspace(self, problem_id: str, holdout_config: dict = None) -> Optional[Path]:
        """
        Create a base workspace for a problem that can be duplicated for multiple agents.
        
        Args:
            problem_id: ID of the problem to create workspace for
            holdout_config: Configuration for holdout test cases
            
        Returns:
            Path to the created base workspace or None if failed
        """
        try:
            # Create base workspace name without agent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_name = f"base_{problem_id}_{timestamp}"
            workspace_path = self.base_dir / workspace_name
            
            # Use direct function call to setup problem
            logger.debug(f"Setting up base workspace for problem {problem_id} in {workspace_path}")
            result = setup_problem_by_id(
                problem_id=problem_id,
                output_dir=str(workspace_path.absolute()),
                release_version=self.release_version,
                verbose=True,
                holdout_config=holdout_config
            )
            
            if result is None:
                logger.error(f"Failed to setup base workspace for {problem_id}")
                return None
            
            # Handle new return format with holdout data
            if len(result) == 3:
                problem_dir, _, holdout_data = result
            else:
                problem_dir, _ = result
                holdout_data = None
            
            actual_workspace = Path(problem_dir)
            
            # Create holdout files one level above the workspace if available
            if holdout_data:
                holdout_dir = actual_workspace.parent
                self._create_holdout_files(holdout_dir, problem_id, holdout_data)
            
            # Verify the required files exist
            required_files = ["problem.md", "solution.py", "test.py", "test_cases.json"]
            for file_name in required_files:
                if not (actual_workspace / file_name).exists():
                    logger.error(f"Required file {file_name} not found in {actual_workspace}")
                    return None
            
            # Store workspace info with special base workspace marker
            self.active_workspaces[actual_workspace] = {
                "problem_id": problem_id,
                "agent_name": "base",  # Special marker for base workspace
                "created_at": datetime.now(),
                "status": "base",
                "workspace_type": "base"
            }
            
            logger.debug(f"Created base workspace: {actual_workspace}")
            return actual_workspace
            
        except Exception as e:
            logger.error(f"Failed to create base workspace: {e}")
            return None
    
    def _create_holdout_files(self, holdout_dir: Path, problem_id: str, holdout_data: dict) -> None:
        """Create holdout files in the specified directory."""
        try:
            # Create holdout test cases file
            holdout_test_file = holdout_dir / f"{problem_id}_test_cases_holdout.json"
            holdout_test_file.write_text(json.dumps(holdout_data['test_cases'], indent=2))
            
            # Create final evaluation script
            eval_script_file = holdout_dir / f"{problem_id}_final_evaluation.py"
            eval_script_file.write_text(holdout_data['evaluation_script'])
            eval_script_file.chmod(0o755)  # Make executable
            
            logger.debug(f"Created holdout files for {problem_id} in {holdout_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create holdout files for {problem_id}: {e}")
    
    def duplicate_workspace(self, base_workspace: Path, agent_name: str) -> Optional[Path]:
        """
        Duplicate a base workspace for a specific agent.
        
        Args:
            base_workspace: Path to the base workspace to duplicate
            agent_name: Name of the agent that will use this workspace
            
        Returns:
            Path to the duplicated workspace or None if failed
        """
        try:
            if not base_workspace.exists():
                logger.error(f"Base workspace does not exist: {base_workspace}")
                return None
            
            # Get problem_id from base workspace info
            base_info = self.active_workspaces.get(base_workspace)
            if not base_info:
                logger.error(f"Base workspace info not found: {base_workspace}")
                return None
            
            problem_id = base_info["problem_id"]
            
            # Create agent-specific workspace name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_name = f"{agent_name}_{problem_id}_{timestamp}"
            agent_workspace = self.base_dir / workspace_name
            
            # Copy the entire base workspace to agent workspace
            logger.debug(f"Duplicating workspace from {base_workspace} to {agent_workspace}")
            shutil.copytree(base_workspace, agent_workspace)
            
            # Store workspace info for the agent workspace
            self.active_workspaces[agent_workspace] = {
                "problem_id": problem_id,
                "agent_name": agent_name,
                "created_at": datetime.now(),
                "status": "active",
                "workspace_type": "agent",
                "base_workspace": str(base_workspace)
            }
            
            logger.debug(f"Created agent workspace: {agent_workspace}")
            return agent_workspace
            
        except Exception as e:
            logger.error(f"Failed to duplicate workspace: {e}")
            return None
    
    def get_workspace_info(self, workspace_path: Path) -> Optional[Dict[str, Any]]:
        """Get information about a workspace."""
        return self.active_workspaces.get(workspace_path)
    
    def list_active_workspaces(self) -> List[Dict[str, Any]]:
        """List all active workspaces."""
        workspaces = []
        for path, info in self.active_workspaces.items():
            workspace_info = info.copy()
            workspace_info["path"] = str(path)
            workspaces.append(workspace_info)
        return workspaces
    
    def cleanup_workspace(self, workspace_path: Path) -> bool:
        """
        Clean up a specific workspace.
        
        Args:
            workspace_path: Path to the workspace to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
                logger.debug(f"Cleaned up workspace: {workspace_path}")
            
            # Remove from active workspaces
            if workspace_path in self.active_workspaces:
                del self.active_workspaces[workspace_path]
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup workspace {workspace_path}: {e}")
            return False
    
    def cleanup_agent_workspaces(self, problem_id: str = None) -> None:
        """
        Clean up agent workspaces, optionally for a specific problem.
        
        Args:
            problem_id: If specified, only clean up agent workspaces for this problem
        """
        if not self.cleanup:
            logger.debug("Cleanup disabled, keeping all workspaces")
            return
        
        workspaces_to_clean = []
        for workspace_path, info in self.active_workspaces.items():
            # Only clean up agent workspaces (not base workspaces)
            if info.get("workspace_type") == "agent":
                if problem_id is None or info.get("problem_id") == problem_id:
                    workspaces_to_clean.append(workspace_path)
        
        for workspace_path in workspaces_to_clean:
            self.cleanup_workspace(workspace_path)
    
    def cleanup_base_workspaces(self) -> None:
        """Clean up all base workspaces."""
        if not self.cleanup:
            logger.debug("Cleanup disabled, keeping all workspaces")
            return
        
        workspaces_to_clean = []
        for workspace_path, info in self.active_workspaces.items():
            # Only clean up base workspaces
            if info.get("workspace_type") == "base":
                workspaces_to_clean.append(workspace_path)
        
        for workspace_path in workspaces_to_clean:
            self.cleanup_workspace(workspace_path)
    
    def cleanup_all(self) -> None:
        """Clean up all active workspaces."""
        if not self.cleanup:
            logger.debug("Cleanup disabled, keeping all workspaces")
            return
            
        workspaces_to_clean = list(self.active_workspaces.keys())
        for workspace_path in workspaces_to_clean:
            self.cleanup_workspace(workspace_path)
    
    def save_workspace_results(self, workspace_path: Path, results_dir: Path) -> bool:
        """
        Save important files from workspace before cleanup.
        
        Args:
            workspace_path: Path to the workspace
            results_dir: Directory to save results to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create results directory
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Files to save
            files_to_save = [
                "solution.py",
                "test_results.json",
                "test.log",
                "agent.log"
            ]
            
            saved_files = []
            for filename in files_to_save:
                src_file = workspace_path / filename
                if src_file.exists():
                    dst_file = results_dir / filename
                    shutil.copy2(src_file, dst_file)
                    saved_files.append(filename)
            
            # Save workspace metadata
            metadata = self.get_workspace_info(workspace_path)
            if metadata:
                metadata["saved_files"] = saved_files
                metadata_file = results_dir / "workspace_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved workspace results to {results_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save workspace results: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all workspaces."""
        self.cleanup_all()


class WorkspaceMonitor:
    """Monitor workspace for changes and test results."""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.solution_file = workspace_path / "solution.py"
        self.test_log = workspace_path / "test.log"
        self.last_solution_mtime = None
        self.last_test_mtime = None
    
    def has_solution_changed(self) -> bool:
        """Check if solution.py has been modified."""
        if not self.solution_file.exists():
            return False
            
        current_mtime = self.solution_file.stat().st_mtime
        if self.last_solution_mtime is None:
            # Initialize baseline and detect if the initial content already looks like a solution
            self.last_solution_mtime = current_mtime
            return self.has_real_solution()
        
        if current_mtime > self.last_solution_mtime:
            self.last_solution_mtime = current_mtime
            # Consider any file modification as a change; tests will determine correctness
            return True
            
        return False
    
    def has_real_solution(self) -> bool:
        """Check if the solution contains actual implementation, not just TODO."""
        content = self.get_solution_content()
        if not content:
            return False
            
        # First, check for obvious TODO markers - if found, it's definitely a template
        todo_markers = ['TODO', 'result = "TODO"', 'print("TODO")', 'return "TODO"', 
                       'TODO: Implement', 'TODO: Replace', '# TODO:']
        if any(marker in content for marker in todo_markers):
            return False  # Templates always have TODO markers
            
        # Check if solution.py is essentially unchanged from template
        # Template characteristics:
        # 1. Has the solve() function with only comments
        # 2. No actual algorithm implementation
        
        # Remove comments and docstrings for analysis
        import re
        # Remove docstrings
        content_no_docstrings = re.sub(r'"""[\s\S]*?"""', '', content)
        content_no_docstrings = re.sub(r"'''[\s\S]*?'''", '', content_no_docstrings)
        # Remove comments
        content_no_comments = re.sub(r'#.*', '', content_no_docstrings)
        
        # Split into lines and filter empty ones
        lines = [line.strip() for line in content_no_comments.split('\n') if line.strip()]
        
        # Count meaningful code lines (not just imports, function definitions, or structural code)
        meaningful_lines = []
        for line in lines:
            # Skip imports, function/class definitions, and structural code
            if (line.startswith('import ') or 
                line.startswith('from ') or
                line.startswith('def ') or
                line.startswith('class ') or
                line.startswith('@') or  # decorators
                line == 'pass' or
                line == 'solve()' or
                line == 'if __name__ == "__main__":' or
                line.startswith('"""') or
                line.startswith("'''")):
                continue
            meaningful_lines.append(line)
        
        # Require at least 1 meaningful line of actual code
        # This is intentionally permissive to avoid missing short but correct solutions
        return len(meaningful_lines) >= 1
    
    def has_test_results(self) -> bool:
        """Check if new test results are available."""
        if not self.test_log.exists():
            return False
            
        current_mtime = self.test_log.stat().st_mtime
        if self.last_test_mtime is None:
            self.last_test_mtime = current_mtime
            return True
        
        if current_mtime > self.last_test_mtime:
            self.last_test_mtime = current_mtime
            return True
            
        return False
    
    def get_solution_content(self) -> Optional[str]:
        """Get the current solution content."""
        if self.solution_file.exists():
            with open(self.solution_file, 'r') as f:
                return f.read()
        return None
    
    def run_tests(self) -> Dict[str, Any]:
        """Run tests and return results."""
        try:
            # Run test.py
            result = subprocess.run(
                ["python", "test.py"],
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                env=build_subprocess_env()
            )
            
            # Parse results
            test_results = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            # Try to parse JSON output if available
            if "test_results.json" in os.listdir(self.workspace_path):
                results_file = self.workspace_path / "test_results.json"
                with open(results_file, 'r') as f:
                    test_results["detailed_results"] = json.load(f)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
