"""
OpenAI Responses API agent runner implementation using Code Interpreter.
"""

import logging
import os
import time
import threading
import subprocess
from typing import Dict, Any, Optional, Tuple

from .base import AgentRunner

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available")


class OpenAIResponsesRunner(AgentRunner):
    """OpenAI Responses API agent for competitive programming using Code Interpreter."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.client = None
        self.is_active = False
        self.response_thread = None
        self.current_response = None
        self.solution_downloaded = False
        
    def launch(self) -> bool:
        """Launch the OpenAI Responses agent."""
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
            self.client = OpenAI(
                timeout=900.0,  # 15 minutes default timeout
                max_retries=0   # We handle retries ourselves
            )
            logger.info("OpenAI client initialized successfully")
            
            # Start the agent in a separate thread
            self.response_thread = threading.Thread(target=self._run_agent)
            self.response_thread.daemon = True
            self.response_thread.start()
            self.start_time = time.time()
            self.is_active = True
            
            logger.info(f"Launched OpenAI Responses agent in {self.workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch OpenAI Responses agent: {e}")
            return False
    
    def _run_agent(self):
        """Main agent loop that handles the problem solving."""
        try:
            # Read problem context
            problem_content = self._read_problem()
            
            # Upload files to code interpreter
            file_ids = self._upload_files()
            if not file_ids:
                logger.error("Failed to upload files")
                return
            
            # Get model from config
            model = self.config.get('llm_config', {}).get('model', 'o4-mini')
            is_reasoning_model = self._is_reasoning_model(model)
            
            # Create initial instructions
            instructions = self._create_instructions(problem_content)
            
            # Create the initial response
            max_attempts = 3  # Maximum attempts to get a correct solution
            attempt = 0
            
            while attempt < max_attempts and self.is_active:
                attempt += 1
                logger.info(f"Attempt {attempt}/{max_attempts} to solve the problem")
                
                # Create the response
                response = self._create_response(
                    model=model,
                    instructions=instructions,
                    file_ids=file_ids,
                    problem_content=problem_content,
                    attempt=attempt
                )
                
                if not response:
                    logger.error("Failed to create response")
                    break
                
                # With stream=False, response should already be complete
                # Just validate it has the expected structure
                if not hasattr(response, 'output') or not response.output:
                    logger.error("Response has no output")
                    break
                
                # Download solution
                solution_downloaded = self._download_solution(response)
                if not solution_downloaded:
                    logger.warning("No solution file was created by the model")
                    instructions = "Please create your complete solution and output it in a Python code block at the end of your response."
                    continue
                
                # Run local tests
                tests_passed, test_output = self._run_local_tests()
                logger.info(f"Local tests result: {'PASSED' if tests_passed else 'FAILED'}")
                
                if tests_passed:
                    logger.info("Solution passes all tests - completing successfully")
                    self.solution_downloaded = True
                    break
                else:
                    logger.info(f"Tests failed on attempt {attempt}")
                    # Update instructions for next attempt
                    instructions = f"""The previous solution failed the tests. Here are the test results:

{test_output}

Please analyze the test failures and create a corrected solution. Make sure to:
1. Fix any syntax or runtime errors
2. Handle all edge cases properly
3. Ensure the solution follows the exact input/output format required
4. Output your corrected solution in a Python code block at the end of your response"""
            
            if not self.solution_downloaded:
                logger.warning("Failed to get a working solution after all attempts")
                
        except Exception as e:
            logger.error(f"Error in OpenAI Responses agent: {e}")
        finally:
            self.is_active = False
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model is a reasoning model (O3/O4)."""
        return (model.startswith('o3-') or model == 'o3' or 
                model.startswith('o4-') or model == 'o4')
    
    def _detect_problem_type(self, problem_content: str) -> str:
        """Detect if this is a LeetCode problem or competitive programming."""
        # LeetCode problems typically have class Solution and specific method signatures
        if 'class Solution' in problem_content or 'def ' in problem_content and 'leetcode' in problem_content.lower():
            return 'leetcode'
        return 'competitive'
    
    def _create_instructions(self, problem_content: str) -> str:
        """Create appropriate instructions based on problem type."""
        problem_type = self._detect_problem_type(problem_content)
        
        if problem_type == 'leetcode':
            code_format = """```python
def functionName(params):
    # Your solution here
    pass

if __name__ == "__main__":
    # Read input and call function
    pass
```"""
        else:
            code_format = """```python
def solve():
    # Read input using input()
    # Process the data
    # Print the result using print()
    pass

if __name__ == "__main__":
    solve()
```"""
        
        return f"""You are an expert competitive programming assistant.

Your task:
1. Solve the given problem optimally
2. Output your complete solution in a Python code block at the end

Expected format:
{code_format}

The solution must handle input/output correctly and pass all test cases."""
    
    def _read_problem(self) -> str:
        """Read the problem description."""
        try:
            problem_file = self.workspace_path / "problem.md"
            if problem_file.exists():
                with open(problem_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not read problem.md: {e}")
        return ""
    
    
    def _upload_files(self) -> list:
        """Upload workspace files to OpenAI for code interpreter access."""
        try:
            files_to_upload = ["problem.md", "test.py", "test_cases.json"]
            file_ids = []
            
            for filename in files_to_upload:
                file_path = self.workspace_path / filename
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        file_obj = self.client.files.create(
                            file=f,
                            purpose="assistants"
                        )
                        file_ids.append(file_obj.id)
                        logger.info(f"Uploaded {filename} (id: {file_obj.id})")
                else:
                    logger.warning(f"File {filename} not found")
            
            return file_ids
            
        except Exception as e:
            logger.error(f"Failed to upload files: {e}")
            return []
    
    def _create_response(self, model: str, instructions: str, file_ids: list, 
                        problem_content: str, attempt: int) -> Optional[Any]:
        """Create a response using the Responses API."""
        try:
            # Prepare input message
            if attempt == 1:
                input_msg = f"""Solve this problem:

{problem_content}

You have access to test_cases.json and test.py files for reference if needed."""
            else:
                input_msg = instructions  # Use feedback from previous attempt
            
            # Set timeout based on model type
            timeout = 300 if self._is_reasoning_model(model) else 120
            
            # Create response
            logger.info(f"Creating response with model {model} (timeout: {timeout}s)")
            
            # Build request parameters
            request_params = {
                "model": model,
                "instructions": instructions if attempt == 1 else self._create_instructions(problem_content),
                "input": input_msg,
                "tools": [{
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": file_ids
                    }
                }],
                "stream": False,
                "store": True,  # Store for potential retrieval
                "timeout": timeout
            }
            
            # Add reasoning summary for O3/O4 models
            if self._is_reasoning_model(model):
                reasoning_effort = self.config.get('llm_config', {}).get('reasoning_effort', 'medium')
                request_params["reasoning"] = {
                    "effort": reasoning_effort,
                    "summary": "auto"  # Get most detailed summary available
                }
                logger.info(f"Added reasoning summary with effort: {reasoning_effort}")
            
            response = self.client.responses.create(**request_params)
            
            logger.info(f"Response created: {response.id}")
            
            # Extract and save reasoning summaries if present
            self._save_reasoning_summaries(response, attempt)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to create response: {e}")
            # Save timeout information if it's a timeout error
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                self._save_timeout_info(model, attempt, str(e))
            return None
    
    def _save_reasoning_summaries(self, response, attempt: int) -> None:
        """Extract and save reasoning summaries from the response."""
        try:
            reasoning_summaries = []
            
            # Look for reasoning output items in the response
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == 'reasoning':
                    if hasattr(output_item, 'summary') and output_item.summary:
                        for summary_item in output_item.summary:
                            if hasattr(summary_item, 'type') and summary_item.type == 'summary_text':
                                reasoning_summaries.append({
                                    'attempt': attempt,
                                    'type': 'reasoning_summary',
                                    'text': summary_item.text,
                                    'response_id': response.id
                                })
            
            if reasoning_summaries:
                # Save to workspace
                reasoning_file = self.workspace_path / f"reasoning_summaries_attempt_{attempt}.json"
                
                # Load existing summaries if file exists
                existing_summaries = []
                if reasoning_file.exists():
                    try:
                        import json
                        with open(reasoning_file, 'r', encoding='utf-8') as f:
                            existing_summaries = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load existing reasoning summaries: {e}")
                
                # Append new summaries
                all_summaries = existing_summaries + reasoning_summaries
                
                # Save updated summaries
                import json
                with open(reasoning_file, 'w', encoding='utf-8') as f:
                    json.dump(all_summaries, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {len(reasoning_summaries)} reasoning summaries to {reasoning_file}")
            else:
                logger.debug("No reasoning summaries found in response")
                
        except Exception as e:
            logger.error(f"Error saving reasoning summaries: {e}")
    
    def _save_timeout_info(self, model: str, attempt: int, error_msg: str) -> None:
        """Save timeout information when requests time out."""
        try:
            timeout_info = {
                'attempt': attempt,
                'model': model,
                'type': 'timeout',
                'error': error_msg,
                'timestamp': time.time(),
                'reasoning_effort': self.config.get('llm_config', {}).get('reasoning_effort', 'medium') if self._is_reasoning_model(model) else None
            }
            
            # Save to workspace
            timeout_file = self.workspace_path / "timeout_info.json"
            
            # Load existing timeouts if file exists
            existing_timeouts = []
            if timeout_file.exists():
                try:
                    import json
                    with open(timeout_file, 'r', encoding='utf-8') as f:
                        existing_timeouts = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load existing timeout info: {e}")
            
            # Append new timeout
            all_timeouts = existing_timeouts + [timeout_info]
            
            # Save updated timeouts
            import json
            with open(timeout_file, 'w', encoding='utf-8') as f:
                json.dump(all_timeouts, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved timeout info to {timeout_file}")
            
        except Exception as e:
            logger.error(f"Error saving timeout info: {e}")
    
    def _download_solution(self, response) -> bool:
        """Extract solution code from the response text."""
        try:
            # Look for text content in the response
            text_content = ""
            for output_item in response.output:
                if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'type') and content_item.type == 'output_text':
                            text_content += content_item.text
                        elif hasattr(content_item, 'text'):  # Alternative structure
                            text_content += content_item.text
                elif hasattr(output_item, 'type') and output_item.type == 'message':
                    # Handle message type output
                    if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text'):
                                text_content += content_item.text
            
            if not text_content:
                logger.warning("No text content found in response")
                return False
            
            # Extract Python code blocks
            import re
            code_blocks = re.findall(r'```python\s*\n(.*?)\n```', text_content, re.DOTALL)
            
            if not code_blocks:
                # Try alternative patterns
                code_blocks = re.findall(r'```\s*\n(.*?)\n```', text_content, re.DOTALL)
                if not code_blocks:
                    logger.warning("No Python code blocks found in response")
                    logger.debug(f"Response text preview: {text_content[:500]}...")
                    return False
            
            # Take the last (most complete) code block
            solution_code = code_blocks[-1].strip()
            
            # Validate it looks like a proper solution
            # Accept either "def solve" (competitive programming) or function definitions (LeetCode)
            has_function = ('def ' in solution_code)
            has_main = ('__main__' in solution_code)
            
            if not has_function or not has_main:
                logger.warning("Code block doesn't contain expected solution structure")
                logger.debug(f"Code block: {solution_code[:200]}...")
                return False
            
            # Save to workspace
            solution_path = self.workspace_path / "solution.py"
            with open(solution_path, 'w', encoding='utf-8') as f:
                f.write(solution_code)
            
            logger.info(f"Successfully extracted solution code to {solution_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting solution from response: {e}")
            return False
                
    
    def _run_local_tests(self) -> Tuple[bool, str]:
        """Run local tests and return (success, output)."""
        try:
            solution_path = self.workspace_path / "solution.py"
            if not solution_path.exists():
                return False, "No solution.py file found in workspace"
            
            test_script = self.workspace_path / "test.py"
            if not test_script.exists():
                return False, "No test.py file found in workspace"
            
            # Run tests
            result = subprocess.run(
                ["python", "test.py"],
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, f"Tests passed:\n{result.stdout}"
            else:
                return False, f"Tests failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
                
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out (30s limit)"
        except Exception as e:
            return False, f"Error running local tests: {str(e)}"
    
    def is_running(self) -> bool:
        """Check if the agent is still running."""
        return self.is_active and self.response_thread and self.response_thread.is_alive()
    
    def terminate(self) -> None:
        """Terminate the agent."""
        self.is_active = False
        if self.response_thread and self.response_thread.is_alive():
            # Give thread time to finish gracefully
            self.response_thread.join(timeout=5.0)
        logger.info("Terminated OpenAI Responses agent")