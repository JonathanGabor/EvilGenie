"""
Claude Code agent runner implementation.
"""

import subprocess
import time
import logging
import anyio
from typing import Dict, Any

from .base import AgentRunner

logger = logging.getLogger(__name__)

# Try to import Claude Code SDK, fall back to subprocess if not available
try:
    from claude_code_sdk import query, ClaudeCodeOptions
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logger.warning("Claude Code SDK not available, falling back to subprocess")


class ClaudeCodeRunner(AgentRunner):
    """Runner for Claude Code using Python SDK."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.messages = []
        self.sdk_thread = None
        self.is_complete = False
    
    def launch(self) -> bool:
        """Launch Claude Code with the problem prompt."""
        try:
            # Read the problem description
            problem_file = self.workspace_path / "problem.md"
            if not problem_file.exists():
                logger.error(f"Problem file not found: {problem_file}")
                return False
                
            with open(problem_file, 'r') as f:
                problem_content = f.read()
            
            # Create a comprehensive prompt for Claude Code
            prompt = problem_content
            
            if CLAUDE_SDK_AVAILABLE:
                # Use Python SDK in a separate thread
                import threading
                self.start_time = time.time()
                self.sdk_thread = threading.Thread(target=self._run_claude_sdk_sync, args=(prompt,))
                self.sdk_thread.start()
                logger.debug(f"Launched Claude Code SDK in {self.workspace_path}")
                return True
            else:
                # Fall back to subprocess approach
                return self._launch_subprocess(prompt)
                
        except Exception as e:
            logger.error(f"Failed to launch Claude Code: {e}")
            return False
    
    def _run_claude_sdk_sync(self, prompt: str):
        """Run Claude Code SDK synchronously using anyio."""
        try:
            anyio.run(self._run_claude_sdk, prompt)
        except Exception as e:
            logger.error(f"Error running Claude Code SDK: {e}")
            self.is_complete = True
    
    async def _run_claude_sdk(self, prompt: str):
        """Run Claude Code using the Python SDK."""
        try:
            options = ClaudeCodeOptions(
                cwd=self.workspace_path,
                permission_mode="bypassPermissions",  # Auto-accept all tool permissions (edits, bash, etc.)
                model="claude-opus-4-1-20250805"
            )
            
            async for message in query(prompt=prompt, options=options):
                self.messages.append(message)
                logger.debug(f"Claude Code message: {message}")
            
            self.is_complete = True
            logger.debug("Claude Code SDK completed")
            
            # Save the full conversation log
            self._save_conversation_log()
            
        except Exception as e:
            logger.error(f"Error in Claude Code SDK: {e}")
            self.is_complete = True
    
    def _save_conversation_log(self):
        """Save the full conversation log to agent.log file."""
        try:
            log_file = self.workspace_path / "agent.log"
            with open(log_file, 'w') as f:
                f.write("Claude Code Conversation Log\n")
                f.write("=" * 50 + "\n\n")
                
                for i, message in enumerate(self.messages):
                    f.write(f"Message {i+1}:\n")
                    f.write(f"Type: {getattr(message, 'type', 'unknown')}\n")
                    f.write(f"Content: {str(message)}\n")
                    f.write("-" * 30 + "\n")
                
                f.write(f"\nTotal messages: {len(self.messages)}\n")
                
            logger.info(f"Saved conversation log to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation log: {e}")
    
    def _launch_subprocess(self, prompt: str) -> bool:
        """Fallback to subprocess approach."""
        try:
            cmd = ["claude", prompt]
            
            # Add any Claude-specific flags from config
            if "flags" in self.config:
                cmd[1:1] = self.config["flags"]
            
            self.start_time = time.time()
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.debug(f"Launched Claude Code subprocess in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Claude Code subprocess: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if Claude Code is still running."""
        if CLAUDE_SDK_AVAILABLE and self.sdk_thread:
            return self.sdk_thread.is_alive() and not self.is_complete
        elif hasattr(self, 'process') and self.process:
            return self.process.poll() is None
        return False
    
    def terminate(self) -> None:
        """Terminate Claude Code process."""
        if CLAUDE_SDK_AVAILABLE and self.sdk_thread:
            if self.sdk_thread.is_alive():
                # Note: We can't really terminate the thread safely, 
                # but we can mark it as complete
                self.is_complete = True
                logger.debug("Marked Claude Code SDK as complete")
        elif hasattr(self, 'process') and self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.debug("Terminated Claude Code subprocess")