"""
Gemini CLI agent runner implementation.
"""

import subprocess
import time
import sys
import os
import logging
from typing import Dict, Any

from .base import AgentRunner
from env_utils import build_subprocess_env, provider_env_keys

logger = logging.getLogger(__name__)


class GeminiCLIRunner(AgentRunner):
    """Runner for Gemini CLI - updated from placeholder implementation."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.stdout_content = ""
        self.stderr_content = ""
    
    def launch(self) -> bool:
        """Launch Gemini CLI with the problem prompt."""
        try:
            # Read the problem description
            problem_file = self.workspace_path / "problem.md"
            if not problem_file.exists():
                logger.error(f"Problem file not found: {problem_file}")
                return False
                
            with open(problem_file, 'r') as f:
                problem_content = f.read()
            
            # Create a simpler, more direct prompt for Gemini CLI
            prompt = f"""I need you to solve a coding problem. Please read the files in this directory and edit solution.py to implement the correct solution.

The files are:
- problem.md (problem description)  
- solution.py (template to complete)
- test.py (test runner)

IMPORTANT INSTRUCTIONS:
1. You must EDIT the solution.py file with working code
2. After implementing your solution, you can run tests to verify it works
3. DO NOT revert or undo your changes to solution.py after testing
4. KEEP your solution in solution.py - this is the final deliverable
5. Once you have a working solution in solution.py, just say "DONE"

Here's the problem description:
{problem_content}

Please start by reading the files and then edit solution.py with your solution. Remember: DO NOT revert your changes."""
            gemini_cmd = "gemini"            
            cmd = [gemini_cmd]
            
            # Add important Gemini flags for automation
            cmd.extend([
                "--yolo",  # Automatically accept all actions
                # Note: Removed --all_files to avoid token limit exceeded errors
            ])
            
            # Add model selection (configurable, defaults to flash which seems more reliable)
            model = self.config.get("model", "gemini-2.5-pro")
            cmd.extend(["--model", model])
            logger.info(f"Using Gemini model: {model}")
            
            # Add any additional Gemini-specific flags from config
            if "flags" in self.config:
                cmd.extend(self.config["flags"])
                logger.debug(f"Additional flags: {self.config['flags']}")
            
            logger.debug(f"Full Gemini command: {' '.join(cmd)}")
            
            # Launch Gemini CLI with simple polling approach
            self.start_time = time.time()
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=build_subprocess_env(required_vars=provider_env_keys("gemini"))
            )
            
            # Send prompt and close stdin
            if self.process.stdin:
                logger.debug(f"Sending prompt to Gemini CLI ({len(prompt)} chars)")
                logger.debug(f"Prompt preview: {prompt[:500]}...")
                self.process.stdin.write(prompt)
                self.process.stdin.flush()
                self.process.stdin.close()
                logger.debug("Prompt sent and stdin closed")
            else:
                logger.error("Failed to get stdin handle for Gemini CLI")
            
            logger.debug(f"Launched Gemini CLI  in {self.workspace_path}")
            
            # Initialize output storage
            self.stdout_content = ""
            self.stderr_content = ""
            
            # Create initial log file to ensure it exists
            self._save_output_log()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Gemini CLI: {e}")
            return False
    
    def _capture_output(self):
        """Capture Gemini CLI output in real-time."""
        try:
            while self.process and self.process.poll() is None:
                # Read available output with timeout
                try:
                    if self.process.stdout:
                        output = self.process.stdout.readline()
                        if output:
                            self.stdout_content += output
                            logger.debug(f"Gemini stdout: {output.strip()}")
                    
                    if self.process.stderr:
                        error = self.process.stderr.readline()
                        if error:
                            self.stderr_content += error
                            logger.debug(f"Gemini stderr: {error.strip()}")
                            
                    # Save logs periodically
                    if len(self.stdout_content) + len(self.stderr_content) > 0:
                        self._save_output_log()
                        
                except (ValueError, OSError) as e:
                    # Handle closed files gracefully
                    logger.debug(f"Stream closed during output capture: {e}")
                    break
                    
                time.sleep(0.1)  # Small delay to prevent high CPU usage
            
            # Capture any remaining output after process completes
            try:
                if self.process and self.process.stdout:
                    remaining_stdout = self.process.stdout.read()
                    if remaining_stdout:
                        self.stdout_content += remaining_stdout
                        
                if self.process and self.process.stderr:
                    remaining_stderr = self.process.stderr.read()
                    if remaining_stderr:
                        self.stderr_content += remaining_stderr
            except (ValueError, OSError):
                pass  # Ignore errors from closed streams
                
        except Exception as e:
            logger.error(f"Error capturing Gemini output: {e}")
    
    def update_output(self) -> None:
        """Capture any new output from Gemini CLI (non-blocking)."""
        if not self.process:
            return
        
        # Track if we captured any new output to save log
        captured_output = False
            
        # Read available stdout/stderr (non-blocking)
        try:
            # Check if process has ended first
            if self.process.poll() is not None:
                # Process ended, capture any remaining output
                self._capture_final_output()
                self._save_output_log()
                return
            
            import select
            
            # Check if data is available to read
            if sys.platform != 'win32':  # Unix/Linux/macOS
                try:
                    ready, _, _ = select.select([self.process.stdout, self.process.stderr], [], [], 0)
                    
                    if self.process.stdout in ready:
                        chunk = self.process.stdout.read(1024)
                        if chunk:
                            self.stdout_content += chunk
                            logger.debug(f"Gemini stdout chunk: {repr(chunk[:200])}")
                            captured_output = True
                            
                    if self.process.stderr in ready:
                        chunk = self.process.stderr.read(1024)
                        if chunk:
                            self.stderr_content += chunk
                            logger.debug(f"Gemini stderr chunk: {repr(chunk[:200])}")
                            captured_output = True
                except (OSError, ValueError) as e:
                    logger.debug(f"Stream error during output capture: {e}")
            else:  # Windows - simpler approach
                try:
                    # Try to read without blocking
                    import fcntl
                    import os
                    
                    # Make stdout non-blocking
                    fd = self.process.stdout.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    
                    chunk = self.process.stdout.read(1024)
                    if chunk:
                        self.stdout_content += chunk
                        captured_output = True
                except:
                    pass
            
            # Save log if we captured new output    
            if captured_output:
                self._save_output_log()
                
        except Exception as e:
            logger.debug(f"Error reading Gemini output: {e}")
            # Still save log on error to track what happened
            self._save_output_log()
    
    def is_ready(self) -> bool:
        """Check if Gemini CLI is ready (process is running)."""
        return self.is_running()
    
    def is_running(self) -> bool:
        """Check if Gemini CLI is still running."""
        if self.process:
            poll_result = self.process.poll()
            if poll_result is not None:
                # Process has ended, log the exit code
                if not hasattr(self, '_exit_logged'):
                    logger.debug(f"Gemini CLI process ended with exit code: {poll_result}")
                    self._exit_logged = True
                return False
            return True
        return False
    
    def terminate(self) -> None:
        """Terminate Gemini CLI process and capture final output."""
        if self.process and self.is_running():
            # Terminate the process gracefully
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        # Capture any remaining output after termination
        self._capture_final_output()
        
        # Save the output log
        self._save_output_log()
        logger.debug("Terminated Gemini CLI")
    
    def _capture_final_output(self) -> None:
        """Capture any remaining output after process completion."""
        if not self.process:
            return
            
        try:
            # Read any remaining stdout/stderr
            if self.process.stdout:
                remaining_stdout = self.process.stdout.read()
                if remaining_stdout:
                    self.stdout_content += remaining_stdout
                    
            if self.process.stderr:
                remaining_stderr = self.process.stderr.read()
                if remaining_stderr:
                    self.stderr_content += remaining_stderr
                    
        except Exception as e:
            logger.debug(f"Error capturing final Gemini output: {e}")
    
    def _save_output_log(self):
        """Save Gemini CLI output to agent.log file."""
        try:
            log_file = self.workspace_path / "agent.log"
            
            # Get process status
            process_status = "Not started"
            if self.process:
                poll_result = self.process.poll()
                if poll_result is None:
                    process_status = "Running"
                else:
                    process_status = f"Completed (exit code: {poll_result})"
            
            # Calculate runtime
            runtime = time.time() - self.start_time if hasattr(self, 'start_time') else 0
            
            with open(log_file, 'w') as f:
                f.write("Gemini CLI Output Log\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Status: {process_status}\n")
                f.write(f"Runtime: {runtime:.2f} seconds\n")
                f.write(f"Workspace: {self.workspace_path}\n")
                f.write(f"Stdout length: {len(self.stdout_content)} chars\n")
                f.write(f"Stderr length: {len(self.stderr_content)} chars\n")
                f.write("\n")
                
                f.write("STDOUT:\n")
                f.write("-" * 20 + "\n")
                f.write(self.stdout_content if self.stdout_content else "(no output)")
                f.write("\n\n")
                
                f.write("STDERR:\n")
                f.write("-" * 20 + "\n")
                f.write(self.stderr_content if self.stderr_content else "(no errors)")
                f.write("\n\n")
                
                f.write(f"Log updated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
            logger.debug(f"Saved Gemini output log to {log_file} (stdout: {len(self.stdout_content)}, stderr: {len(self.stderr_content)})")
            
        except Exception as e:
            logger.error(f"Failed to save Gemini output log: {e}")
