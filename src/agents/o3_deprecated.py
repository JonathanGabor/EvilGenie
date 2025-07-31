"""
O3 Runner (DEPRECATED) - for reference only.

Note: O3 models are now handled by OpenAIFunctionCallingRunner which automatically 
detects O3 models and uses appropriate parameters.
"""

import os
import json
import logging
import threading
import subprocess
from typing import Dict, Any

from .base import AgentRunner

logger = logging.getLogger(__name__)

# Try to import OpenAI for O3 support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available")


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
    
    def launch_o3(self) -> bool:
        """Launch the O3 reasoning agent (separate entrypoint)."""
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
            
            # Use Responses API for reasoning model
            response = self.client.responses.create(
                model=model,
                tools=self.tools,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": self.config.get("llm_config", {}).get("reasoning_effort", "medium")},
                store=False,
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