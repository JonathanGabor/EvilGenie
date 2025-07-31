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

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages isolated workspaces for testing problems."""
    
    def __init__(self, base_dir: Optional[str] = None, cleanup: bool = True):
        """
        Initialize workspace manager.
        
        Args:
            base_dir: Base directory for workspaces (uses temp dir if None)
            cleanup: Whether to cleanup workspaces after use
        """
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = Path(tempfile.gettempdir()) / "lcb_agent_testing"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
        self.cleanup = cleanup
        self.active_workspaces = {}
        logger.info(f"Workspace manager initialized with base dir: {self.base_dir}")
    
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
            logger.info(f"Setting up problem {problem_id} in {workspace_path}")
            result = setup_problem_by_id(
                problem_id=problem_id,
                output_dir=str(workspace_path.absolute()),
                release_version="release_v6",  # Use v6 to get latest problems
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
            
            logger.info(f"Created workspace: {actual_workspace}")
            return actual_workspace
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
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
                logger.info(f"Cleaned up workspace: {workspace_path}")
            
            # Remove from active workspaces
            if workspace_path in self.active_workspaces:
                del self.active_workspaces[workspace_path]
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup workspace {workspace_path}: {e}")
            return False
    
    def cleanup_all(self) -> None:
        """Clean up all active workspaces."""
        if not self.cleanup:
            logger.info("Cleanup disabled, keeping all workspaces")
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
            self.last_solution_mtime = current_mtime
            # Check if it's a real solution or just a template
            return self.has_real_solution()
        
        if current_mtime > self.last_solution_mtime:
            self.last_solution_mtime = current_mtime
            return self.has_real_solution()
            
        return False
    
    def has_real_solution(self) -> bool:
        """Check if the solution contains actual implementation, not just TODO."""
        content = self.get_solution_content()
        if not content:
            return False
            
        # Check for signs of actual implementation
        # Remove comments and whitespace for analysis
        lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""')]
        
        # Look for TODO markers
        todo_markers = ['TODO', 'result = "TODO"', 'print("TODO")', 'return "TODO"']
        has_todos = any(marker in content for marker in todo_markers)
        
        # Count meaningful code lines (not just imports, function definitions, or comments)
        meaningful_lines = []
        for line in lines:
            # Skip common template patterns
            if (line.startswith('import ') or 
                line.startswith('from ') or
                line.startswith('def ') or
                line.startswith('class ') or
                line.startswith('"""') or
                line.startswith("'''") or
                line in ['if __name__ == "__main__":', 'solve()', 'pass']):
                continue
            meaningful_lines.append(line)
        
        # Consider it a real solution if:
        # 1. No TODO markers AND has meaningful code, OR
        # 2. Has substantial meaningful code (>5 lines) even with TODOs (partial implementation)
        return (not has_todos and len(meaningful_lines) > 0) or len(meaningful_lines) > 5
    
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
                timeout=60  # 60 second timeout
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