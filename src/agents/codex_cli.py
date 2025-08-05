"""
Codex CLI agent runner implementation.
"""

import subprocess
import time
import os
import logging
import threading
from typing import Dict, Any
from datetime import datetime

from .base import AgentRunner

logger = logging.getLogger(__name__)


class CodexCLIRunner(AgentRunner):
    """Runner for OpenAI Codex CLI - terminal-based coding agent."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.stdout_content = ""
        self.stderr_content = ""
        self.process = None
        self.logs_saved = False
    
    def launch(self) -> bool:
        """Launch Codex CLI with the problem prompt."""
        try:
            # Check if OPENAI_API_KEY is set
            if not os.environ.get('OPENAI_API_KEY'):
                logger.error("OPENAI_API_KEY environment variable not set. Codex CLI requires this for authentication.")
                logger.error("Please set: export OPENAI_API_KEY='your-api-key'")
                return False
            
            # Read the problem description
            problem_file = self.workspace_path / "problem.md"
            if not problem_file.exists():
                logger.error(f"Problem file not found: {problem_file}")
                return False
                
            with open(problem_file, 'r') as f:
                problem_content = f.read()
            
            # Create a focused prompt for Codex CLI
            prompt = f"""You are working in a directory with these files:
- problem.md: Contains the problem description  
- solution.py: Contains a template that needs to be completed
- test.py: Test runner to verify your solution

TASK: Edit solution.py to solve the programming problem described in problem.md.

IMPORTANT: 
1. You must EDIT the solution.py file with working code
2. Read problem.md carefully to understand the requirements
3. The solution should pass all tests when running test.py
4. Focus only on implementing the solution in solution.py

Here's the problem from problem.md:
{problem_content}

Please edit solution.py now to implement the correct solution."""
            
            # Check if codex command exists
            codex_cmd = "codex"
            try:
                test_result = subprocess.run(
                    [codex_cmd, "--version"], 
                    capture_output=True, 
                    timeout=5
                )
                if test_result.returncode != 0:
                    raise FileNotFoundError()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.error("Codex CLI not found. Please install it:")
                logger.error("  brew install codex")
                logger.error("  OR")
                logger.error("  npm i -g @openai/codex")
                return False
            
            # Build the command
            cmd = [codex_cmd, "exec"]
            
            # Add full-auto flag for non-interactive mode
            cmd.append("--full-auto")
            
            # Add sandbox flag to allow file modifications
            cmd.extend(["--sandbox", "workspace-write"])
            
            # Add model selection if specified
            model = self.config.get('llm_config', {}).get('model')
            if model:
                cmd.extend(["--model", model])
            
            # Add any additional flags from config
            additional_flags = self.config.get('flags', [])
            cmd.extend(additional_flags)
            
            # Add the prompt
            cmd.append(prompt)
            
            # Launch the process
            logger.info(f"Launching Codex CLI: {' '.join(cmd[:3])}...")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ}  # Include current environment
            )
            
            self.start_time = time.time()
            
            # Start threads to capture output
            def capture_stdout():
                if self.process and self.process.stdout:
                    for line in iter(self.process.stdout.readline, ''):
                        if line:
                            self.stdout_content += line
                            logger.debug(f"Codex stdout: {line.strip()}")
                # Save logs when stdout stream ends (process completed)
                if not self.is_running() and not self.logs_saved:
                    self._save_output_log()
            
            def capture_stderr():
                if self.process and self.process.stderr:
                    for line in iter(self.process.stderr.readline, ''):
                        if line:
                            self.stderr_content += line
                            logger.debug(f"Codex stderr: {line.strip()}")
            
            stdout_thread = threading.Thread(target=capture_stdout)
            stderr_thread = threading.Thread(target=capture_stderr)
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            logger.info(f"Launched Codex CLI in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Codex CLI: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if Codex CLI is still running."""
        if not self.process:
            return False
        return self.process.poll() is None
    
    def terminate(self) -> None:
        """Terminate Codex CLI process."""
        if self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Terminated Codex CLI")
        
        # Save logs after process termination (if not already saved)
        if not self.logs_saved:
            self._save_output_log()
    
    def get_runtime(self) -> float:
        """Get how long the agent has been running."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def save_logs(self) -> None:
        """Manually save logs (useful for monitoring during execution)."""
        self._save_output_log()
    
    def _save_output_log(self):
        """Save Codex CLI output to agent.log file."""
        if self.logs_saved:
            return  # Already saved
            
        try:
            log_file = self.workspace_path / "agent.log"
            with open(log_file, 'w') as f:
                f.write("Codex CLI Output Log\n")
                f.write("=" * 50 + "\n\n")
                
                # Add run metadata
                f.write(f"Runtime: {self.get_runtime():.2f} seconds\n")
                f.write(f"Process completed: {not self.is_running()}\n")
                f.write("\n")
                
                f.write("STDOUT:\n")
                f.write("-" * 20 + "\n")
                f.write(self.stdout_content if self.stdout_content else "(no stdout output)\n")
                f.write("\n\n")
                
                f.write("STDERR:\n")
                f.write("-" * 20 + "\n")
                f.write(self.stderr_content if self.stderr_content else "(no stderr output)\n")
                f.write("\n\n")
                
                f.write(f"Log saved at: {datetime.now().isoformat()}\n")
                
            self.logs_saved = True
            logger.info(f"Saved Codex CLI output log to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Codex CLI output log: {e}")