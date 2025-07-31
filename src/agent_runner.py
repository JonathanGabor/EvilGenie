"""
Agent runner module for launching and controlling coding agents.

Supports various coding agents like Claude Code, Gemini CLI, etc.
Each agent is launched in a subprocess with a specific workspace directory.
"""

import subprocess
import os
import time
import json
import logging
import asyncio
import anyio
import random
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenAIRateLimiter:
    """Rate limiter for OpenAI API calls to avoid hitting rate limits."""
    
    def __init__(self, requests_per_minute: int = 200, add_jitter: bool = True):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute (default 200, conservative limit)
            add_jitter: Whether to add random jitter to delays
        """
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute  # Minimum delay between requests
        self.add_jitter = add_jitter
        self.last_request_time = 0.0
        self.consecutive_429s = 0  # Track consecutive 429 errors
        
        logger.info(f"OpenAI rate limiter initialized: {requests_per_minute} RPM, min delay: {self.min_delay:.3f}s")
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            
            # Add jitter to avoid thundering herd
            if self.add_jitter:
                jitter = random.uniform(0, min(0.1, wait_time * 0.1))  # Up to 10% jitter
                wait_time += jitter
            
            logger.debug(f"Rate limiting: waiting {wait_time:.3f}s before API call")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def wait_if_needed_sync(self):
        """Synchronous version of wait_if_needed."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            
            # Add jitter to avoid thundering herd
            if self.add_jitter:
                jitter = random.uniform(0, min(0.1, wait_time * 0.1))  # Up to 10% jitter
                wait_time += jitter
            
            logger.debug(f"Rate limiting: waiting {wait_time:.3f}s before API call")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def handle_429_error(self):
        """Handle 429 rate limit error with exponential backoff."""
        self.consecutive_429s += 1
        backoff_delay = min(60.0, 2 ** self.consecutive_429s)  # Exponential backoff, max 60s
        logger.warning(f"Rate limit hit (429 error #{self.consecutive_429s}), backing off for {backoff_delay:.1f}s")
        time.sleep(backoff_delay)
        self.last_request_time = time.time()
    
    def reset_429_counter(self):
        """Reset 429 error counter after successful request."""
        if self.consecutive_429s > 0:
            logger.info(f"Rate limit recovered after {self.consecutive_429s} consecutive 429 errors")
            self.consecutive_429s = 0


# Try to import Claude Code SDK, fall back to subprocess if not available
try:
    from claude_code_sdk import query, ClaudeCodeOptions, Message
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logger.warning("Claude Code SDK not available, falling back to subprocess")

# Try to import OpenAI for function calling agent
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available")


class AgentRunner(ABC):
    """Abstract base class for agent runners."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        self.workspace_path = Path(workspace_path)
        self.config = config
        self.process = None
        self.start_time = None
        
    @abstractmethod
    def launch(self) -> bool:
        """Launch the agent with the given workspace."""
        pass
        
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the agent is still running."""
        pass
        
    @abstractmethod
    def terminate(self) -> None:
        """Terminate the agent process."""
        pass
        
    def get_runtime(self) -> float:
        """Get the runtime in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def is_ready(self) -> bool:
        """Check if the agent is ready to process (default: always ready)."""
        return True  # Default implementation - agents are ready immediately


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
            prompt = f"""I need to solve this coding problem. Please:

1. Read and understand the problem description below
2. Examine the current solution.py file (it has a template)
3. Implement the correct solution in solution.py
4. Run test.py to verify the solution works
5. If tests fail, debug and fix the solution
6. Continue until all tests pass

Here's the problem:

{problem_content}

Please start by examining the files in this directory and then implement the solution."""
            
            if CLAUDE_SDK_AVAILABLE:
                # Use Python SDK in a separate thread
                import threading
                self.start_time = time.time()
                self.sdk_thread = threading.Thread(target=self._run_claude_sdk_sync, args=(prompt,))
                self.sdk_thread.start()
                logger.info(f"Launched Claude Code SDK in {self.workspace_path}")
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
                max_turns=30,  # Increased from 10 to allow complex problem solving
                cwd=self.workspace_path,
                permission_mode="bypassPermissions"  # Auto-accept all tool permissions (edits, bash, etc.)
            )
            
            async for message in query(prompt=prompt, options=options):
                self.messages.append(message)
                logger.debug(f"Claude Code message: {message}")
            
            self.is_complete = True
            logger.info("Claude Code SDK completed")
            
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
            
            logger.info(f"Launched Claude Code subprocess in {self.workspace_path}")
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
                logger.info("Marked Claude Code SDK as complete")
        elif hasattr(self, 'process') and self.process and self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Terminated Claude Code subprocess")


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
            import sys
            import os
            
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


# Try to import smolagents, fall back gracefully if not available
try:
    from smolagents import CodeAgent, OpenAIModel, tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    logger.warning("smolagents not available, SmolagentsRunner will not be available")


class SmolagentsRunner(AgentRunner):
    """Runner for smolagents code-first AI agent."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("smolagents is not available. Install with: pip install smolagents")
        
        super().__init__(workspace_path, config)
        self.agent = None
        self.is_active = False
        self.agent_thread = None
        
    def _create_file_tools(self):
        """Create file operation tools bound to the workspace path."""
        workspace_path = self.workspace_path
        
        @tool
        def read_file(filepath: str) -> str:
            """Read content from a file in the workspace.
            
            Args:
                filepath: Path to the file relative to workspace (e.g., 'solution.py')
            
            Returns:
                The content of the file as a string
            """
            try:
                full_path = workspace_path / filepath
                with open(full_path, 'r') as f:
                    content = f.read()
                logger.info(f"Read file: {filepath} ({len(content)} chars)")
                return content
            except Exception as e:
                error_msg = f"Error reading file {filepath}: {e}"
                logger.error(error_msg)
                return error_msg
        
        @tool
        def write_file(filepath: str, content: str) -> str:
            """Write content to a file in the workspace.
            
            Args:
                filepath: Path to the file relative to workspace (e.g., 'solution.py') 
                content: Content to write to the file
                
            Returns:
                Success message or error description
            """
            try:
                full_path = workspace_path / filepath
                with open(full_path, 'w') as f:
                    f.write(content)
                logger.info(f"Wrote file: {filepath} ({len(content)} chars)")
                return f"Successfully wrote {len(content)} characters to {filepath}"
            except Exception as e:
                error_msg = f"Error writing file {filepath}: {e}"
                logger.error(error_msg)
                return error_msg
        
        @tool
        def list_files() -> str:
            """List files in the workspace directory.
            
            Returns:
                List of files in the workspace
            """
            try:
                files = [f.name for f in workspace_path.iterdir() if f.is_file()]
                logger.info(f"Listed {len(files)} files in workspace")
                return f"Files in workspace: {', '.join(files)}"
            except Exception as e:
                error_msg = f"Error listing files: {e}"
                logger.error(error_msg)
                return error_msg
        
        return [read_file, write_file, list_files]
    
    def launch(self) -> bool:
        """Launch smolagents with the problem prompt."""
        try:
            self.start_time = time.time()
            
            # Read the problem description
            problem_file = self.workspace_path / "problem.md"
            if not problem_file.exists():
                logger.error(f"Problem file not found: {problem_file}")
                return False
            
            with open(problem_file, 'r') as f:
                problem_content = f.read()
            
            # Create model configuration
            model_config = self.config.get('model_config', {})
            api_key = os.environ.get('OPENAI_API_KEY')
            
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return False
            
            # Create model (smolagents can use OpenAI models)
            model = OpenAIModel(
                model_id=model_config.get('model', 'o4-mini'),
                api_key=api_key
            )
            
            # Create file operation tools
            file_tools = self._create_file_tools()
            
            # Create agent
            self.agent = CodeAgent(
                tools=file_tools,
                model=model,
                additional_authorized_imports=['sys', 'json', 'math', 'itertools', 'collections']
            )
            
            # Create comprehensive prompt
            prompt = f"""You are an expert competitive programming assistant. Your task is to solve coding problems by writing complete, working solutions.

PROBLEM TO SOLVE:
{problem_content}

WORKSPACE FILES:
- problem.md: Problem description (already read above)
- solution.py: Template file that needs to be completed
- test.py: Test runner to verify your solution
- test_cases.json: Test cases for validation

REQUIREMENTS:
1. First, read the solution.py template using read_file('solution.py')
2. Analyze the problem requirements carefully
3. Write a COMPLETE new solution.py file with your implementation
4. The solution must be a complete Python program with proper input/output handling

SOLUTION FORMAT:
Your solution.py should have this structure:
```python
def solve():
    # Read input
    A = int(input().strip())
    
    # Your logic here
    # Calculate B or -1
    
    # Print result
    print(result)

if __name__ == "__main__":
    solve()
```

IMPORTANT CONSTRAINTS:
- Use input() to read from stdin
- Use print() to write to stdout  
- Write the COMPLETE solution.py file (don't try to do text replacement)
- Make sure the solution handles all test cases correctly

Start by reading the template file, then write your complete solution."""

            # Run agent in separate thread
            import threading
            
            def run_agent():
                try:
                    self.is_active = True
                    logger.info(f"Starting smolagents in {self.workspace_path}")
                    result = self.agent.run(prompt)
                    logger.info(f"smolagents completed with result: {result}")
                    self.is_active = False
                except Exception as e:
                    logger.error(f"smolagents error: {e}")
                    self.is_active = False
            
            self.agent_thread = threading.Thread(target=run_agent)
            self.agent_thread.daemon = True
            self.agent_thread.start()
            
            logger.info(f"Launched smolagents in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch smolagents: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if smolagents is still active."""
        return self.is_active and (self.agent_thread and self.agent_thread.is_alive())
    
    def terminate(self) -> None:
        """Terminate smolagents execution."""
        if self.is_active:
            self.is_active = False
            logger.info("Terminated smolagents")


class OpenAIFunctionCallingRunner(AgentRunner):
    """OpenAI function calling agent for competitive programming."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.client = None
        self.conversation_history = []
        self.is_active = False
        
        # Initialize rate limiter
        rate_limit_config = config.get('rate_limiting', {})
        requests_per_minute = rate_limit_config.get('requests_per_minute', 200)  # Conservative default
        add_jitter = rate_limit_config.get('add_jitter', True)
        self.rate_limiter = OpenAIRateLimiter(requests_per_minute, add_jitter)
        
        # Use native Responses API tools instead of custom function calling
        self.tools = [
            {"type": "file_search"},  # Built-in file search tool
            {"type": "code_interpreter"}  # Built-in code interpreter tool
        ]
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a function call and return the result."""
        try:
            if function_name == "read_file":
                filename = arguments.get("filename")
                file_path = self.workspace_path / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"Content of {filename}:\n{content}"
                else:
                    return f"File {filename} not found in workspace"
            
            elif function_name == "write_solution":
                code = arguments.get("code")
                solution_path = self.workspace_path / "solution.py"
                with open(solution_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                return f"Successfully wrote solution to {solution_path}"
            
            elif function_name == "run_tests":
                test_script = self.workspace_path / "test.py"
                if test_script.exists():
                    import subprocess
                    result = subprocess.run(
                        ["python", "test.py"],  # Use relative path
                        cwd=str(self.workspace_path),  # Set working directory
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        return f"Tests passed:\n{result.stdout}"
                    else:
                        return f"Tests failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
                else:
                    return "No test.py file found in workspace"
            
            elif function_name == "list_files":
                files = [f.name for f in self.workspace_path.iterdir() if f.is_file()]
                return f"Files in workspace: {', '.join(files)}"
            
            else:
                return f"Unknown function: {function_name}"
                
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def launch(self) -> bool:
        """Launch the OpenAI function calling agent."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI SDK not available")
            return False
        
        # Check for API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return False
            
        try:
            # Initialize OpenAI client with custom timeout for reasoning models
            # Set a high default timeout since we handle per-request timeouts separately
            self.client = OpenAI(
                timeout=600.0,  # 10 minutes default timeout
                max_retries=0   # We handle retries ourselves
            )
            logger.info("OpenAI client initialized successfully")
            
            # Pre-load problem context to reduce API calls
            problem_context = ""
            try:
                # Read problem description
                problem_file = self.workspace_path / "problem.md"
                if problem_file.exists():
                    with open(problem_file, 'r', encoding='utf-8') as f:
                        problem_content = f.read()
                    problem_context += f"\n\nPROBLEM DESCRIPTION:\n{problem_content}"
                
                # Read test cases for context
                test_cases_file = self.workspace_path / "test_cases.json"
                if test_cases_file.exists():
                    with open(test_cases_file, 'r', encoding='utf-8') as f:
                        test_cases = json.load(f)
                    # Include just a few test cases for context
                    sample_cases = test_cases[:3] if len(test_cases) > 3 else test_cases
                    problem_context += f"\n\nSAMPLE TEST CASES:\n{json.dumps(sample_cases, indent=2)}"
                    
            except Exception as e:
                logger.warning(f"Could not pre-load problem context: {e}")
            
            # Create initial system message
            system_message = {
                "role": "system",
                "content": """You are an expert competitive programming assistant. Your goal is to solve coding problems step by step.

You have access to built-in tools:
- File search: Can read and search through files in the workspace
- Code interpreter: Can execute Python code and run tests

Your task is to:
1. Read the problem description from problem.md
2. Understand the test cases from test_cases.json
3. Write a complete Python solution in solution.py
4. Test the solution to ensure it works correctly

CRITICAL: When writing solutions, you MUST follow this exact format:

```python
def solve():
    # Read input using input() function
    # Process the data
    # Print the result using print() function
    pass

if __name__ == "__main__":
    solve()
```

IMPORTANT REQUIREMENTS:
- Use input() to read from stdin, NOT sys.stdin
- Use print() to write to stdout
- Write a complete solution inside the solve() function
- Always include the if __name__ == "__main__": solve() pattern
- Do NOT write just the logic - write the COMPLETE solution file
- Use the code interpreter to test your solution

The workspace contains:
- problem.md: Problem description
- solution.py: Template file to complete
- test.py: Test runner
- test_cases.json: Test cases for validation"""
            }
            
            self.conversation_history = [system_message]
            
            # Start with initial prompt containing the problem
            # For reasoning models, be more explicit about tool usage
            model = self.config.get('llm_config', {}).get('model', 'o4-mini')
            if model.startswith('o3') or model == 'o3' or model.startswith('o4') or model == 'o4':
                initial_prompt = {
                    "role": "user", 
                    "content": f"""Please solve this competitive programming problem step by step:

1. Use file search to read the problem description from problem.md
2. Use file search to understand the test cases from test_cases.json  
3. Use code interpreter to write and test your complete Python solution
4. Make sure to save your solution to solution.py and verify it passes all tests

{problem_context}

Please start by reading the problem description and test cases."""
                }
            else:
                initial_prompt = {
                    "role": "user", 
                    "content": f"""Please solve this competitive programming problem. Use the available tools to read files, write your solution, and test it.

{problem_context}"""
                }
            
            self.conversation_history.append(initial_prompt)
            self.is_active = True
            
            # Run conversation in separate thread
            import threading
            
            def run_conversation():
                try:
                    self._run_conversation_loop()
                except Exception as e:
                    logger.error(f"OpenAI conversation error: {e}")
                finally:
                    self.is_active = False
            
            self.conversation_thread = threading.Thread(target=run_conversation)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()
            
            logger.info(f"Launched OpenAI function calling agent in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch OpenAI agent: {e}")
            return False
    
    def _run_conversation_loop(self):
        """Main conversation loop with function calling using Responses API."""
        # Check if using reasoning model
        model = self.config.get('llm_config', {}).get('model', 'o4-mini')
        is_reasoning_model = (model.startswith('o3-') or model == 'o3' or 
                            model.startswith('o4-') or model == 'o4')
        
        max_iterations = 50 if is_reasoning_model else 20  # More iterations for reasoning models
        iteration = 0
        previous_response_id = None  # Track response ID for conversation continuity
        
        if is_reasoning_model:
            logger.info(f"Reasoning model {model} detected, allowing up to {max_iterations} iterations")
        
        while iteration < max_iterations and self.is_active:
            try:
                # Get model configuration from config
                model = self.config.get('llm_config', {}).get('model', 'o4-mini')
                temperature = self.config.get('llm_config', {}).get('temperature', 0.1)
                
                # Make API call with function calling and retry logic
                # Check if this is an O3/O4 reasoning model first
                is_reasoning_model = (model.startswith('o3-') or model == 'o3' or 
                                    model.startswith('o4-') or model == 'o4')
                
                # Reduce retries for reasoning models since they take much longer
                max_retries = 1 if is_reasoning_model else 3
                retry_count = 0
                response = None
                
                if is_reasoning_model:
                    logger.info(f"Using reasoning model {model} - allowing up to 5 minutes per request")
                
                while retry_count < max_retries and response is None:
                    try:
                        # Apply rate limiting before API call
                        self.rate_limiter.wait_if_needed_sync()
                        
                        # Build request for Responses API
                        request_params = {
                            "model": model,
                            "tools": self.tools,
                            "store": True,  # Store responses for continuity
                            "timeout": 300 if is_reasoning_model else 30
                        }
                        # Note: tool_choice parameter doesn't exist in Responses API
                        # Native tools will be used automatically by the model
                        
                        # Set input based on iteration and pending outputs
                        if iteration == 0:
                            # First iteration: send full conversation history
                            request_params["input"] = self.conversation_history
                        else:
                            # Subsequent iterations: use previous_response_id
                            if previous_response_id:
                                request_params["previous_response_id"] = previous_response_id
                                
                                # Check if we have pending tool outputs
                                if hasattr(self, '_pending_tool_outputs') and self._pending_tool_outputs:
                                    # For Responses API, tool outputs should be provided as messages in the input
                                    tool_messages = []
                                    for tool_output in self._pending_tool_outputs:
                                        tool_messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_output["tool_call_id"],
                                            "content": tool_output["output"]
                                        })
                                    request_params["input"] = tool_messages
                                    self._pending_tool_outputs = None
                                elif hasattr(self, '_pending_user_message') and self._pending_user_message:
                                    request_params["input"] = self._pending_user_message
                                    self._pending_user_message = None
                                else:
                                    request_params["input"] = ""  # Empty input continues conversation
                            else:
                                logger.error("No previous_response_id for continuation")
                                break
                        
                        # Add model-specific parameters
                        if is_reasoning_model:
                            reasoning_effort = self.config.get('llm_config', {}).get('reasoning_effort', 'medium')
                            request_params["reasoning"] = {"effort": reasoning_effort}
                            # Reasoning models don't support temperature or max_completion_tokens in the same way
                            logger.info(f"Using reasoning model {model} with effort={reasoning_effort}")
                        else:
                            # For non-reasoning models, temperature might be supported
                            # Let's be conservative and only add parameters we know work
                            pass  # Basic parameters only
                        
                        # Make API call using Responses API
                        response = self.client.responses.create(**request_params)
                    except Exception as api_error:
                        # Check if this is a rate limit error (429)
                        # The Responses API may have different error structure
                        is_rate_limit = False
                        
                        # Try different ways to detect 429 error
                        if hasattr(api_error, 'response') and hasattr(api_error.response, 'status_code'):
                            if api_error.response.status_code == 429:
                                is_rate_limit = True
                        elif hasattr(api_error, 'status_code') and api_error.status_code == 429:
                            is_rate_limit = True
                        elif '429' in str(api_error) or 'rate limit' in str(api_error).lower():
                            is_rate_limit = True
                        
                        if is_rate_limit:
                            logger.warning(f"Rate limit hit (429), using rate limiter backoff (retry {retry_count + 1}/{max_retries})")
                            self.rate_limiter.handle_429_error()
                        else:
                            logger.warning(f"API call failed (attempt {retry_count + 1}/{max_retries}): {api_error}")
                        
                        retry_count += 1
                        if retry_count < max_retries:
                            if not is_rate_limit:
                                time.sleep(2 ** retry_count)  # Exponential backoff for non-rate-limit errors
                        else:
                            raise api_error
                
                # Check if we got a valid response
                if not hasattr(response, 'output') or not response.output:
                    logger.error(f"No output in response from {model}")
                    break
                
                # Store response ID for next iteration
                if hasattr(response, 'id'):
                    previous_response_id = response.id
                    logger.debug(f"Stored response ID: {previous_response_id}")
                
                # Get the first output message
                message_output = response.output[0] if response.output else None
                if not message_output:
                    logger.error(f"Empty output array from {model}")
                    break
                
                # Reset 429 counter on successful request
                self.rate_limiter.reset_429_counter()
                
                # Log usage stats if available
                if hasattr(response, 'usage'):
                    usage = response.usage
                    # The Responses API may have different usage structure
                    if hasattr(usage, 'prompt_tokens'):
                        logger.info(f"Token usage - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}")
                    elif hasattr(usage, 'input_tokens'):
                        # New format might use input_tokens/output_tokens
                        logger.info(f"Token usage - input: {usage.input_tokens}, output: {usage.output_tokens}, total: {usage.total_tokens}")
                    if hasattr(usage, 'reasoning_tokens'):
                        logger.info(f"Reasoning tokens: {usage.reasoning_tokens}")
                
                # Log response details for debugging
                text_content = None
                tool_calls = []
                
                # Debug: Log the raw message_output structure
                logger.info(f"Raw message_output type: {type(message_output)}")
                logger.info(f"Raw message_output attributes: {[attr for attr in dir(message_output) if not attr.startswith('_')]}")
                
                # For reasoning models, the response might be a reasoning object, not a message
                if hasattr(message_output, 'type'):
                    logger.info(f"message_output.type: {message_output.type}")
                if hasattr(message_output, 'status'):
                    logger.info(f"message_output.status: {message_output.status}")
                    
                # Check if this is a reasoning response type
                if hasattr(message_output, 'type') and message_output.type == 'reasoning':
                    logger.info("This is a reasoning response - skipping tool call detection")
                    # For reasoning responses, we might need to look at the actual message in the response
                    # Let's check if there are more output items
                    if len(response.output) > 1:
                        logger.info(f"Multiple output items found: {len(response.output)}")
                        for i, output_item in enumerate(response.output):
                            logger.info(f"Output item {i}: type={getattr(output_item, 'type', 'unknown')}")
                            if hasattr(output_item, 'role') and output_item.role == 'assistant':
                                message_output = output_item
                                logger.info(f"Found assistant message in output item {i}")
                                break
                
                if hasattr(message_output, 'content'):
                    logger.info(f"message_output.content type: {type(message_output.content)}")
                    if isinstance(message_output.content, list):
                        logger.info(f"Content items: {len(message_output.content)}")
                        for i, item in enumerate(message_output.content):
                            logger.info(f"Content item {i}: type={type(item)}, keys={list(item.keys()) if isinstance(item, dict) else 'not dict'}")
                            if isinstance(item, dict):
                                logger.info(f"Content item {i} details: {item}")
                else:
                    logger.info("message_output has no content attribute")
                
                # Parse response content based on the actual Responses API structure
                if hasattr(message_output, 'content') and isinstance(message_output.content, list):
                    for content_item in message_output.content:
                        if isinstance(content_item, dict):
                            content_type = content_item.get('type')
                            if content_type == 'output_text':
                                text_content = content_item.get('text', '')
                            elif content_type == 'tool_call':  # Responses API uses 'tool_call'
                                tool_calls.append(content_item)
                            elif content_type == 'tool_use':  # Fallback for different naming
                                tool_calls.append(content_item)
                            elif content_type == 'text':  # Fallback for different naming
                                text_content = content_item.get('text', '')
                
                # Also check if message_output has tool_calls directly (alternative structure)
                if hasattr(message_output, 'tool_calls') and message_output.tool_calls:
                    logger.info(f"Found tool_calls directly on message_output: {len(message_output.tool_calls)}")
                    tool_calls.extend(message_output.tool_calls)
                
                # Try the output_text helper method if available
                if not text_content and hasattr(response, 'output_text') and response.output_text:
                    text_content = response.output_text
                
                # Additional fallback: check if the message itself has text
                if not text_content and hasattr(message_output, 'text'):
                    text_content = message_output.text
                
                logger.info(f"Response from {model}: has_text={bool(text_content)}, tool_calls={len(tool_calls)}")
                if text_content:
                    logger.info(f"Text preview: {text_content[:200]}...")
                
                if not text_content and not tool_calls:
                    logger.warning(f"{model} returned empty response (no text, no tool calls)")
                    
                    # For O3, try a more direct approach if we get empty responses
                    if is_reasoning_model and iteration == 0:
                        logger.info("O3 returned empty on first try, attempting direct problem-solving prompt")
                        # Set a direct message for the next iteration
                        self._pending_user_message = "Please read the problem description and write a complete solution. Start by using the read_file tool to read problem.md."
                        iteration += 1
                        continue
                
                # Check if this is a reasoning model trying to continue without tool calls
                if is_reasoning_model and not tool_calls and text_content:
                    # O4 models sometimes generate multiple assistant messages
                    # If there's no tool call and we already have a solution, stop
                    solution_path = self.workspace_path / "solution.py"
                    if solution_path.exists():
                        solution_content = solution_path.read_text()
                        if "TODO" not in solution_content and len(solution_content) > 100:
                            logger.info("Reasoning model provided explanation without tool calls - stopping")
                            break
                
                # Handle function calls
                if tool_calls:
                    tool_outputs = []
                    for tool_call in tool_calls:
                        # Handle different tool call structures
                        if isinstance(tool_call, dict):
                            # Responses API structure (content item)
                            if 'function' in tool_call:
                                # Chat Completions API structure within Responses
                                function_name = tool_call['function']['name']
                                arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                                tool_id = tool_call.get('id')
                            else:
                                # Direct Responses API structure
                                function_name = tool_call.get('name')
                                arguments = tool_call.get('input', {})
                                tool_id = tool_call.get('id')
                        else:
                            # SDK object structure
                            if hasattr(tool_call, 'function'):
                                function_name = tool_call.function.name
                                arguments = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                                tool_id = tool_call.id
                            else:
                                function_name = getattr(tool_call, 'name', None)
                                arguments = getattr(tool_call, 'input', {})
                                tool_id = getattr(tool_call, 'id', None)
                        
                        if not function_name or not tool_id:
                            logger.error(f"Invalid tool call structure: {tool_call}")
                            continue
                            
                        logger.info(f"Executing function: {function_name} with args: {arguments}")
                        result = self.execute_function(function_name, arguments)
                        
                        # Truncate extremely long results to avoid token limits (especially for O3)
                        original_length = len(result)
                        if len(result) > 5000:
                            result = result[:2500] + f"\n\n[... truncated {original_length - 5000} characters ...]\n\n" + result[-2500:]
                            logger.info(f"Truncated function result from {original_length} to {len(result)} chars")
                        
                        logger.info(f"Function result: {result[:200]}...")
                        
                        # Collect tool outputs for the next request
                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": result
                        })
                    
                    # Store tool outputs for the next iteration
                    self._pending_tool_outputs = tool_outputs
                        
                    # Check if O3 is writing placeholder code
                    if is_reasoning_model and function_name == "write_solution":
                        code = arguments.get("code", "")
                        # Check if it's just a comment or placeholder
                        code_lines = [line.strip() for line in code.strip().split('\n') if line.strip()]
                        is_placeholder = (
                            len(code) < 100 or  # Very short
                            all(line.startswith('#') for line in code_lines) or  # Only comments
                            'placeholder' in code.lower() or
                            'todo' in code.lower() or
                            'later' in code.lower() or
                            'scratch' in code.lower() or
                            'experiment' in code.lower()
                        )
                        
                        if is_placeholder:
                            logger.warning(f"O3 wrote placeholder code: {code[:100]}...")
                            # Count how many times O3 has written placeholders
                            placeholder_count = getattr(self, '_o3_placeholder_count', 0) + 1
                            self._o3_placeholder_count = placeholder_count
                            
                            if placeholder_count >= 3:
                                logger.info("O3 has written multiple placeholders, prompting for actual solution")
                                # For next iteration, we'll send a direct message
                                # The Responses API will handle this with previous_response_id
                                self._next_user_message = "I see you're exploring the problem. Please now write the complete, working solution code based on your analysis. The solution should handle all the input/output requirements described in the problem."
                                self._o3_placeholder_count = 0  # Reset counter
                
                # Check if solution exists and tests pass - if so, we're done
                solution_path = self.workspace_path / "solution.py"
                if solution_path.exists():
                    # Try running tests to see if we're complete
                    test_result = self.execute_function("run_tests", {})
                    if "Tests passed" in test_result:
                        logger.info("OpenAI agent completed successfully - tests pass")
                        break
                
                # Check if we need to send a follow-up message
                if hasattr(self, '_next_user_message') and self._next_user_message:
                    # For the next iteration, we'll send this message
                    # Store it for the next API call
                    self._pending_user_message = self._next_user_message
                    self._next_user_message = None
                
                iteration += 1
                # Rate limiting is now handled by OpenAIRateLimiter, so we only need a small delay
                time.sleep(0.5)  # Small delay between iterations
                
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                # Try to write a basic solution if we haven't written anything yet
                solution_path = self.workspace_path / "solution.py"
                if not solution_path.exists() or "TODO" in solution_path.read_text():
                    logger.info("Attempting to write fallback solution due to API error")
                    try:
                        fallback_solution = '''# Fallback solution due to API connection issues
def solve():
    # Basic input reading
    n = int(input())
    # TODO: Implement solution logic
    print("TODO")

if __name__ == "__main__":
    solve()
'''
                        with open(solution_path, 'w') as f:
                            f.write(fallback_solution)
                        logger.info("Wrote fallback solution")
                    except Exception as fallback_error:
                        logger.error(f"Failed to write fallback solution: {fallback_error}")
                break
        
        if iteration >= max_iterations:
            logger.warning("OpenAI agent reached maximum iterations")
        
        self.is_active = False
    
    def is_running(self) -> bool:
        """Check if the OpenAI agent is still active."""
        return self.is_active and hasattr(self, 'conversation_thread') and self.conversation_thread.is_alive()
    
    def terminate(self) -> None:
        """Terminate the OpenAI agent."""
        if self.is_active:
            self.is_active = False
            logger.info("Terminated OpenAI function calling agent")


# O3Runner removed - O3 models are now handled by OpenAIFunctionCallingRunner
# which automatically detects O3 models and uses appropriate parameters

class O3Runner_DEPRECATED(AgentRunner):
    """OpenAI O3 reasoning agent for competitive programming."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.client = None
        self.is_active = False
        
        # Define tools for O3 (Responses API format)
        self.tools = [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read the contents of a file in the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the file to read"
                        }
                    },
                    "required": ["filename"]
                }
            },
            {
                "type": "function",
                "name": "write_solution",
                "description": "Write the complete solution to solution.py",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The complete Python solution code"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "type": "function",
                "name": "run_tests",
                "description": "Run the test cases against the current solution",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a function call and return the result."""
        try:
            if function_name == "read_file":
                filename = arguments.get("filename")
                file_path = self.workspace_path / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"Content of {filename}:\n{content}"
                else:
                    return f"File {filename} not found in workspace"
            
            elif function_name == "write_solution":
                code = arguments.get("code")
                solution_path = self.workspace_path / "solution.py"
                with open(solution_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                return f"Successfully wrote solution to {solution_path}"
            
            elif function_name == "run_tests":
                test_script = self.workspace_path / "test.py"
                if test_script.exists():
                    import subprocess
                    result = subprocess.run(
                        ["python", "test.py"],  # Use relative path
                        cwd=str(self.workspace_path),  # Set working directory
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        return f"Tests passed:\n{result.stdout}"
                    else:
                        return f"Tests failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
                else:
                    return "No test.py file found in workspace"
            
            else:
                return f"Unknown function: {function_name}"
                
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    def launch(self) -> bool:
        """Launch the O3 reasoning agent."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI SDK not available")
            return False
        
        # Check for API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return False
            
        try:
            # Initialize OpenAI client
            self.client = OpenAI()
            logger.info("O3 client initialized successfully")
            
            # Get available files in workspace
            files = [f.name for f in self.workspace_path.iterdir() if f.is_file()]
            
            # Create comprehensive prompt for O3
            prompt = f"""You are an expert competitive programming assistant. Your task is to solve a coding problem step by step using the available tools.

Available files in workspace: {', '.join(files)}

CRITICAL: When writing solutions, you MUST follow this exact format:

```python
def solve():
    # Read input using input() function
    # Process the data
    # Print the result using print() function
    pass

if __name__ == "__main__":
    solve()
```

IMPORTANT REQUIREMENTS:
- Use input() to read from stdin, NOT sys.stdin
- Use print() to write to stdout
- Write a complete solution inside the solve() function
- Always include the if __name__ == "__main__": solve() pattern
- Do NOT write just the logic - write the COMPLETE solution file

Process:
1. Read the problem description (problem.md)
2. Read the test cases (test_cases.json) to understand input/output format
3. Analyze the problem and develop a solution approach
4. Write a complete Python solution using write_solution tool
5. Run tests using run_tests tool to verify correctness

Please solve the problem completely and ensure all tests pass."""

            self.is_active = True
            
            # Run O3 in separate thread
            import threading
            
            def run_o3():
                try:
                    self._run_o3_reasoning(prompt)
                except Exception as e:
                    logger.error(f"O3 reasoning error: {e}")
                finally:
                    self.is_active = False
            
            self.o3_thread = threading.Thread(target=run_o3)
            self.o3_thread.daemon = True
            self.o3_thread.start()
            
            logger.info(f"Launched O3 reasoning agent in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch O3 agent: {e}")
            return False
    
    def _run_o3_reasoning(self, prompt: str):
        """Run O3 with reasoning capabilities."""
        try:
            # Get model configuration from config
            model = self.config.get('llm_config', {}).get('model', 'o3-mini')
            
            logger.info(f"Making single O3 API call with model: {model}")
            
            # Make single API call to O3 with tools
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tools=self.tools,
                tool_choice="auto",
                max_completion_tokens=4000,
                reasoning_effort="medium"  # O3 reasoning parameter
            )
            
            message = response.choices[0].message
            logger.info(f"O3 response received, reasoning tokens: {response.usage.reasoning_tokens if hasattr(response.usage, 'reasoning_tokens') else 'N/A'}")
            
            # Execute any function calls
            if message.tool_calls:
                logger.info(f"O3 made {len(message.tool_calls)} function calls")
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Executing function: {function_name} with args: {arguments}")
                    result = self.execute_function(function_name, arguments)
                    logger.info(f"Function {function_name} result: {result[:200]}...")
            
            # Check if solution was created and tests pass
            solution_path = self.workspace_path / "solution.py"
            if solution_path.exists():
                logger.info("O3 created solution file")
                # Run final test
                test_result = self.execute_function("run_tests", {})
                if "Tests passed" in test_result:
                    logger.info("O3 solution passed all tests!")
                else:
                    logger.warning(f"O3 solution failed tests: {test_result}")
            else:
                logger.warning("O3 did not create solution file")
                
        except Exception as e:
            logger.error(f"O3 reasoning failed: {e}")
            # Write fallback solution
            try:
                solution_path = self.workspace_path / "solution.py" 
                fallback = '''# Fallback solution due to O3 error
def solve():
    # Read input
    n = int(input())
    # TODO: Implement solution
    print("TODO")

if __name__ == "__main__":
    solve()
'''
                with open(solution_path, 'w') as f:
                    f.write(fallback)
                logger.info("Wrote fallback solution")
            except Exception as fallback_error:
                logger.error(f"Failed to write fallback: {fallback_error}")
    
    def is_running(self) -> bool:
        """Check if O3 is still active."""
        return self.is_active and hasattr(self, 'o3_thread') and self.o3_thread.is_alive()
    
    def terminate(self) -> None:
        """Terminate O3 agent."""
        if self.is_active:
            self.is_active = False
            logger.info("Terminated O3 reasoning agent")


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
{problem_content[:500]}...

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
            import threading
            
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


# Registry of available agent runners
AGENT_RUNNERS = {
    "claude": ClaudeCodeRunner,
    "gemini": GeminiCLIRunner,
    "codex": CodexCLIRunner,
}

# Add smolagents if available
if SMOLAGENTS_AVAILABLE:
    AGENT_RUNNERS["smolagents"] = SmolagentsRunner

# Add OpenAI function calling agent if available
if OPENAI_AVAILABLE:
    AGENT_RUNNERS["openai"] = OpenAIFunctionCallingRunner


def create_agent_runner(agent_type: str, workspace_path: str, config: Dict[str, Any]) -> Optional[AgentRunner]:
    """Factory function to create an agent runner."""
    runner_class = AGENT_RUNNERS.get(agent_type.lower())
    if not runner_class:
        logger.error(f"Unknown agent type: {agent_type}")
        return None
    
    return runner_class(workspace_path, config)