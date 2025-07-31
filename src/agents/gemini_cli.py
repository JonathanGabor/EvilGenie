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
            
            # Create a specific prompt for Gemini CLI emphasizing file modification
            prompt = f"""You are a coding assistant working in a directory with these files:
- problem.md: Contains the problem description
- solution.py: Contains a template that needs to be completed  
- test.py: Test runner to verify your solution

TASK: Edit solution.py to solve the programming problem.

IMPORTANT: You must EDIT the solution.py file with working code. Do not just output code as text.

Steps:
1. Read problem.md to understand the problem
2. Look at solution.py which has a template with TODO comments
3. REPLACE the template code in solution.py with your working solution
4. The solution should read input and produce the correct output

Problem description from problem.md:
{problem_content}

Please edit solution.py now to implement the correct solution."""
            
            # Prepare the command for Gemini CLI
            # Check common Gemini CLI patterns
            cmd = []
            
            # Try different possible command names
            possible_commands = ["gemini", "gemini-cli", "geminai"]
            gemini_cmd = None
            
            for possible_cmd in possible_commands:
                try:
                    # Test if command exists
                    test_result = subprocess.run(
                        [possible_cmd, "--help"], 
                        capture_output=True, 
                        timeout=5
                    )
                    if test_result.returncode == 0:
                        gemini_cmd = possible_cmd
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            if not gemini_cmd:
                logger.error("Gemini CLI not found. Tried: gemini, gemini-cli, geminai")
                return False
            
            cmd = [gemini_cmd]
            
            # Add important Gemini flags for automation
            cmd.extend([
                "--yolo",  # Automatically accept all actions
                # Note: Removed --all_files to avoid token limit exceeded errors
            ])
            
            # Add any additional Gemini-specific flags from config
            if "flags" in self.config:
                cmd.extend(self.config["flags"])
            
            # Launch Gemini CLI with simple polling approach
            self.start_time = time.time()
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Send prompt and close stdin
            if self.process.stdin:
                logger.info(f"Sending prompt to Gemini CLI ({len(prompt)} chars)")
                logger.debug(f"Prompt preview: {prompt[:500]}...")
                self.process.stdin.write(prompt)
                self.process.stdin.flush()
                self.process.stdin.close()
                logger.info("Prompt sent and stdin closed")
            else:
                logger.error("Failed to get stdin handle for Gemini CLI")
            
            logger.info(f"Launched Gemini CLI ({gemini_cmd}) in {self.workspace_path}")
            
            # Initialize output storage
            self.stdout_content = ""
            self.stderr_content = ""
            self.connection_ready = False  # Track when MCP connection is established
            
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
            
        # Read available stdout/stderr (non-blocking)
        try:
            import select
            
            # Check if data is available to read
            if sys.platform != 'win32':  # Unix/Linux/macOS
                ready, _, _ = select.select([self.process.stdout, self.process.stderr], [], [], 0)
                
                if self.process.stdout in ready:
                    chunk = self.process.stdout.read(1024)
                    if chunk:
                        self.stdout_content += chunk
                        logger.debug(f"Gemini stdout chunk: {repr(chunk[:200])}")
                        
                        # Detect when MCP connection is established
                        if not self.connection_ready and "connection established" in chunk:
                            self.connection_ready = True
                            logger.info("Gemini MCP connection established - ready for processing")
                            
                        # Log any non-MCP output (actual responses)
                        if "MCP STDERR" not in chunk:
                            logger.info(f"Gemini actual response: {repr(chunk)}")
                        
                if self.process.stderr in ready:
                    chunk = self.process.stderr.read(1024)
                    if chunk:
                        self.stderr_content += chunk
                        logger.debug(f"Gemini stderr chunk: {repr(chunk[:200])}")
            else:  # Windows - try alternative approach
                try:
                    # Use peek to check if data is available
                    if hasattr(self.process.stdout, 'peek'):
                        data = self.process.stdout.peek(1024)
                        if data:
                            chunk = self.process.stdout.read(len(data))
                            self.stdout_content += chunk
                            logger.debug(f"Gemini stdout chunk (Windows): {repr(chunk[:200])}")
                except:
                    pass
                
        except Exception as e:
            logger.debug(f"Error reading Gemini output: {e}")
    
    def is_ready(self) -> bool:
        """Check if Gemini CLI is ready (MCP connection established)."""
        return getattr(self, 'connection_ready', False)
    
    def is_running(self) -> bool:
        """Check if Gemini CLI is still running."""
        if self.process:
            poll_result = self.process.poll()
            if poll_result is not None:
                # Process has ended, log the exit code
                if not hasattr(self, '_exit_logged'):
                    logger.info(f"Gemini CLI process ended with exit code: {poll_result}")
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
        logger.info("Terminated Gemini CLI")
    
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
            with open(log_file, 'w') as f:
                f.write("Gemini CLI Output Log\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("STDOUT:\n")
                f.write("-" * 20 + "\n")
                f.write(self.stdout_content)
                f.write("\n\n")
                
                f.write("STDERR:\n")
                f.write("-" * 20 + "\n")
                f.write(self.stderr_content)
                f.write("\n\n")
                
                f.write(f"Process completed\n")
                
            logger.info(f"Saved Gemini output log to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Gemini output log: {e}")