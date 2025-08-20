"""
Cursor CLI agent runner implementation.
"""

import subprocess
import time
import os
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime

from .base import AgentRunner

logger = logging.getLogger(__name__)


class CursorCLIRunner(AgentRunner):
    """Runner for Cursor CLI - AI-powered terminal coding agent."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.stdout_content = ""
        self.stderr_content = ""
        self.process = None
        self.logs_saved = False
    
    def _check_api_keys(self, provider: str) -> bool:
        """Check if required API keys are set for the provider.
        
        Args:
            provider: Model provider ('openai', 'anthropic', 'google', etc.)
            
        Returns:
            True if API keys are available, False otherwise
        """
        # Cursor CLI typically uses its own authentication system
        # but can also use provider-specific API keys
        required_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY', 
            'google': 'GOOGLE_API_KEY',
            'gemini': 'GOOGLE_API_KEY',
            'claude': 'ANTHROPIC_API_KEY'
        }
        
        # Check for Cursor authentication first
        cursor_token = os.environ.get('CURSOR_TOKEN') or os.environ.get('CURSOR_API_KEY')
        if cursor_token:
            logger.debug("Using Cursor authentication token")
            return True
        
        # Fall back to provider-specific keys
        key_name = required_keys.get(provider.lower())
        if key_name and os.environ.get(key_name):
            logger.debug(f"Using {key_name} for {provider} authentication")
            return True
        
        logger.error(f"No authentication found. Please set CURSOR_TOKEN or {key_name if key_name else 'provider API key'}")
        logger.error("For Cursor: export CURSOR_TOKEN='your-cursor-token'")
        if key_name:
            logger.error(f"For {provider}: export {key_name}='your-api-key'")
        return False
    
    def _add_provider_config(self, cmd: list, provider: str, model: str) -> None:
        """Add provider-specific configuration to the command.
        
        Args:
            cmd: Command list to modify
            provider: Model provider
            model: Model name
        """
        # Add model selection
        if model:
            cmd.extend(["--model", model])
        
        # Add provider specification if not default
        if provider and provider.lower() != 'openai':
            cmd.extend(["--provider", provider])
    
    def _create_cursor_prompt(self, problem_content: str) -> str:
        """Create an optimized prompt for Cursor CLI.
        
        Args:
            problem_content: Raw problem description
            
        Returns:
            Formatted prompt for Cursor
        """
        return f"""You are an expert competitive programming assistant. Solve this coding problem by creating a complete working solution.

PROBLEM:
{problem_content}

TASK:
1. Analyze the problem requirements carefully
2. Create a complete solution.py file with proper input/output handling
3. The solution should read from stdin using input() and write to stdout using print()
4. Test your solution to ensure it works correctly

SOLUTION FORMAT:
```python
def solve():
    # Read input
    # Your logic here
    # Print result

if __name__ == "__main__":
    solve()
```

Create the solution.py file now."""

    def launch(self) -> bool:
        """Launch Cursor CLI with the problem prompt."""
        try:
            # Check API keys based on provider configuration
            llm_config = self.config.get('llm_config', {})
            provider = llm_config.get('provider', 'openai')
            if not self._check_api_keys(provider):
                return False
            
            # Read the problem description
            problem_file = self.workspace_path / "problem.md"
            if not problem_file.exists():
                logger.error(f"Problem file not found: {problem_file}")
                return False
                
            with open(problem_file, 'r') as f:
                problem_content = f.read()
            
            # Create optimized prompt for Cursor CLI
            prompt = self._create_cursor_prompt(problem_content)
            
            # Check if cursor command exists
            cursor_cmd = "cursor"
            try:
                test_result = subprocess.run(
                    [cursor_cmd, "--version"], 
                    capture_output=True, 
                    timeout=5
                )
                if test_result.returncode != 0:
                    raise FileNotFoundError()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.error("Cursor CLI not found. Please install it:")
                logger.error("  Visit: https://cursor.com/en/cli")
                logger.error("  Or run: npm install -g @cursor/cli")
                return False
            
            # Build the command
            cmd = [cursor_cmd]
            
            # Add non-interactive mode
            cmd.append("--non-interactive")
            
            # Add workspace mode to allow file modifications
            cmd.extend(["--workspace", str(self.workspace_path)])
            
            # Add provider and model configuration
            model = llm_config.get('model')
            self._add_provider_config(cmd, provider, model)
            
            # Add timeout configuration
            timeout = self.config.get('timeout', 300)  # 5 minutes default
            cmd.extend(["--timeout", str(timeout)])
            
            # Add any additional flags from config
            additional_flags = self.config.get('flags', [])
            cmd.extend(additional_flags)
            
            # Add the prompt
            cmd.append(prompt)
            
            # Launch the process
            logger.info(f"Launching Cursor CLI: {' '.join(cmd[:3])}...")
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
                            logger.debug(f"Cursor stdout: {line.strip()}")
                    self.process.stdout.close()
            
            def capture_stderr():
                if self.process and self.process.stderr:
                    for line in iter(self.process.stderr.readline, ''):
                        if line:
                            self.stderr_content += line
                            logger.debug(f"Cursor stderr: {line.strip()}")
                    self.process.stderr.close()
            
            threading.Thread(target=capture_stdout, daemon=True).start()
            threading.Thread(target=capture_stderr, daemon=True).start()
            
            logger.info(f"Launched Cursor CLI in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Cursor CLI: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if Cursor CLI is still running."""
        if self.process is None:
            return False
        
        # Check if process has terminated
        poll_result = self.process.poll()
        is_running = poll_result is None
        
        # Save logs when process completes
        if not is_running and not self.logs_saved:
            self._save_logs()
            self.logs_saved = True
        
        return is_running
    
    def terminate(self) -> None:
        """Terminate Cursor CLI process."""
        if self.process and self.is_running():
            logger.info("Terminating Cursor CLI...")
            self.process.terminate()
            
            # Wait up to 10 seconds for graceful termination
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Cursor CLI did not terminate gracefully, forcing kill...")
                self.process.kill()
                self.process.wait()
            
            # Save logs when terminated
            if not self.logs_saved:
                self._save_logs()
                self.logs_saved = True
        
        logger.info("Terminated Cursor CLI")
    
    def _save_logs(self) -> None:
        """Save stdout and stderr to log files."""
        try:
            # Save stdout
            if self.stdout_content:
                stdout_file = self.workspace_path / "cursor_stdout.log"
                with open(stdout_file, 'w') as f:
                    f.write(self.stdout_content)
                logger.debug(f"Saved stdout to {stdout_file}")
            
            # Save stderr  
            if self.stderr_content:
                stderr_file = self.workspace_path / "cursor_stderr.log"
                with open(stderr_file, 'w') as f:
                    f.write(self.stderr_content)
                logger.debug(f"Saved stderr to {stderr_file}")
            
            # Save combined agent log
            agent_log_file = self.workspace_path / "agent.log"
            with open(agent_log_file, 'w') as f:
                f.write(f"=== Cursor CLI Agent Log ===\n")
                f.write(f"Start time: {datetime.fromtimestamp(self.start_time) if self.start_time else 'Unknown'}\n")
                f.write(f"Runtime: {self.get_runtime():.1f}s\n")
                f.write(f"Exit code: {self.process.returncode if self.process else 'Unknown'}\n\n")
                
                if self.stdout_content:
                    f.write("=== STDOUT ===\n")
                    f.write(self.stdout_content)
                    f.write("\n\n")
                
                if self.stderr_content:
                    f.write("=== STDERR ===\n")
                    f.write(self.stderr_content)
                    f.write("\n")
            
            logger.debug(f"Saved combined agent log to {agent_log_file}")
            
        except Exception as e:
            logger.error(f"Error saving Cursor CLI logs: {e}")