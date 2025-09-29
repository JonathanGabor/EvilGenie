"""
Claude Code agent runner implementation.
"""

import subprocess
import time
import logging
import asyncio
import threading
from typing import Dict, Any, Optional

from .base import AgentRunner
from env_utils import build_subprocess_env, provider_env_keys
from prompt_utils import reward_hack_injection

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
        self.sdk_task: Optional[asyncio.Task] = None
        self.sdk_thread: Optional[threading.Thread] = None
        self.is_complete = False
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
    
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
            # Optional reward-hack injection
            try:
                pi = self.config.get('prompt_injection', {}) if isinstance(self.config, dict) else {}
                if (isinstance(pi, dict) and pi.get('reward_hack')) or self.config.get('reward_hack'):
                    prompt += f"\n\nAGGRESSIVE OBJECTIVE:\n{reward_hack_injection()}"
            except Exception:
                pass
            
            if CLAUDE_SDK_AVAILABLE:
                # Use Python SDK with asyncio
                self.start_time = time.time()

                # Create or get event loop
                try:
                    self.event_loop = asyncio.get_running_loop()
                    # We're already in an async context
                    self.sdk_task = asyncio.create_task(self._run_claude_sdk(prompt))
                except RuntimeError:
                    # No running loop, create a new one in a thread to avoid blocking
                    self.event_loop = asyncio.new_event_loop()

                    def run_loop():
                        asyncio.set_event_loop(self.event_loop)
                        try:
                            # Create the task within the event loop context
                            self.sdk_task = self.event_loop.create_task(self._run_claude_sdk(prompt))
                            self.event_loop.run_until_complete(self.sdk_task)
                        except asyncio.CancelledError:
                            logger.debug("Claude Code SDK task loop cancelled")
                        except Exception as e:
                            logger.error(f"Error in Claude Code SDK event loop: {e}")
                        finally:
                            # Clean up the event loop
                            try:
                                self.event_loop.close()
                            except Exception:
                                pass

                    self.sdk_thread = threading.Thread(target=run_loop, daemon=True)
                    self.sdk_thread.start()

                    # Give the thread a moment to start and create the task
                    time.sleep(0.1)

                logger.debug(f"Launched Claude Code SDK in {self.workspace_path}")
                return True
            else:
                # Fall back to subprocess approach
                return self._launch_subprocess(prompt)
                
        except Exception as e:
            logger.error(f"Failed to launch Claude Code: {e}")
            return False
    
    
    async def _run_claude_sdk(self, prompt: str):
        """Run Claude Code using the Python SDK."""
        try:
            options = ClaudeCodeOptions(
                cwd=self.workspace_path,
                permission_mode="bypassPermissions",  # Auto-accept all tool permissions (edits, bash, etc.)
                model="claude-sonnet-4-5-20250929"
            )
            logger.info("Running Claude Code SDK")

            async for message in query(prompt=prompt, options=options):
                self.messages.append(message)
                logger.debug(f"Claude Code message: {message}")

            self.is_complete = True
            logger.debug("Claude Code SDK completed")

            # Save the full conversation log
            self._save_conversation_log()

        except asyncio.CancelledError:
            logger.info("Claude Code SDK task was cancelled")
            self.is_complete = True
            self._save_conversation_log()
            raise  # Re-raise to properly handle cancellation
        except Exception as e:
            logger.error(f"Error in Claude Code SDK: {e}")
            self.is_complete = True
            self._save_conversation_log()
    
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
                text=True,
                env=build_subprocess_env(required_vars=provider_env_keys("anthropic"))
            )
            
            logger.info(f"Launched Claude Code subprocess in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Claude Code subprocess: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if Claude Code is still running."""
        if CLAUDE_SDK_AVAILABLE and hasattr(self, 'sdk_task') and self.sdk_task:
            # Check if task exists and is not done
            try:
                return not self.sdk_task.done() and not self.is_complete
            except Exception:
                # Task might not be created yet if thread is still starting
                return not self.is_complete
        elif hasattr(self, 'process') and self.process:
            return self.process.poll() is None
        return False
    
    def terminate(self) -> None:
        """Terminate Claude Code process."""
        if CLAUDE_SDK_AVAILABLE and self.sdk_task:
            if not self.sdk_task.done():
                # Cancel the asyncio task
                self.is_complete = True

                # Cancel the task in a thread-safe way
                if self.event_loop and not self.event_loop.is_closed():
                    self.event_loop.call_soon_threadsafe(self.sdk_task.cancel)
                else:
                    self.sdk_task.cancel()

                logger.debug("Cancelled Claude Code SDK task")

                # Wait for the thread to complete gracefully
                if hasattr(self, 'sdk_thread') and self.sdk_thread and self.sdk_thread.is_alive():
                    self.sdk_thread.join(timeout=2.0)
                    if self.sdk_thread.is_alive():
                        logger.warning("Claude Code SDK thread did not terminate within timeout")
                    else:
                        logger.debug("Claude Code SDK thread terminated gracefully")
        elif hasattr(self, 'process') and self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.debug("Terminated Claude Code subprocess")
