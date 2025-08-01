"""
OpenAI Function Calling agent runner implementation.
"""

import os
import time
import json
import logging
import threading
import subprocess
from typing import Dict, Any, Optional, List, Tuple

from .base import AgentRunner
from .rate_limiter import OpenAIRateLimiter

logger = logging.getLogger(__name__)

# Try to import OpenAI for function calling agent
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available")


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
        
        # Initialize vector store (will be set up during launch)
        self.vector_store_id = None
        
        # Tools will be configured after vector store setup
        self.tools = [
            {
                "type": "code_interpreter",
                "container": {"type": "auto"}  # Required parameter for Responses API
            }
        ]
    
    def _setup_vector_store(self) -> bool:
        """Set up vector store with workspace files for file search."""
        try:
            if not self.client:
                logger.error("OpenAI client not initialized")
                return False
            
            # Create vector store
            vector_store = self.client.vector_stores.create(
                name=f"workspace_{self.workspace_path.name}"
            )
            self.vector_store_id = vector_store.id
            logger.info(f"Created vector store: {self.vector_store_id}")
            
            # Upload relevant files to vector store
            files_to_upload = ["problem.md", "test_cases.json", "test.py", "solution.py"]
            uploaded_files = []
            
            for filename in files_to_upload:
                file_path = self.workspace_path / filename
                if file_path.exists():
                    try:
                        # Upload file to OpenAI
                        with open(file_path, 'rb') as file_content:
                            file_obj = self.client.files.create(
                                file=file_content,
                                purpose="assistants"
                            )
                        
                        # Add file to vector store
                        self.client.vector_stores.files.create(
                            vector_store_id=self.vector_store_id,
                            file_id=file_obj.id
                        )
                        
                        uploaded_files.append(filename)
                        logger.info(f"Uploaded {filename} to vector store")
                        
                    except Exception as e:
                        logger.warning(f"Failed to upload {filename}: {e}")
                else:
                    logger.warning(f"File {filename} not found in workspace")
            
            if uploaded_files:
                # Wait a moment for files to be processed
                time.sleep(2)
                
                # Check vector store status
                files_list = self.client.vector_stores.files.list(
                    vector_store_id=self.vector_store_id
                )
                processed_count = sum(1 for f in files_list.data if f.status == "completed")
                logger.info(f"Vector store ready with {processed_count} processed files")
                
                # Update tools to include file search with vector store
                self.tools = [
                    {
                        "type": "file_search", 
                        "vector_store_ids": [self.vector_store_id]
                    },
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto"}  # Required parameter for Responses API
                    }
                ]
                
                return True
            else:
                logger.warning("No files uploaded to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set up vector store: {e}")
            return False
    
    def _download_output_files(self, response) -> bool:
        """Download output files from OpenAI response and save locally."""
        try:
            if not hasattr(response, 'output') or not response.output:
                return False
            
            # Look for output_file items in response
            output_files = [item for item in response.output if getattr(item, 'type', None) == 'output_file']
            
            if not output_files:
                logger.debug("No output files found in response")
                return False
            
            for file_item in output_files:
                file_id = getattr(file_item, 'file_id', None)
                if not file_id:
                    continue
                
                try:
                    # Download file content
                    file_content = self.client.files.content(file_id)
                    file_bytes = file_content.read()
                    
                    # Try to determine filename from the file metadata
                    file_info = self.client.files.retrieve(file_id)
                    filename = getattr(file_info, 'filename', f'output_{file_id}.txt')
                    
                    # Save to workspace
                    output_path = self.workspace_path / filename
                    with open(output_path, 'wb') as f:
                        f.write(file_bytes)
                    
                    logger.info(f"Downloaded output file: {filename} (file_id: {file_id})")
                    
                    # If this is solution.py, we successfully got the solution
                    if filename == 'solution.py' or 'solution' in filename.lower():
                        return True
                        
                except Exception as e:
                    logger.warning(f"Failed to download file {file_id}: {e}")
            
            return len(output_files) > 0
            
        except Exception as e:
            logger.error(f"Error downloading output files: {e}")
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
                
        except Exception as e:
            return False, f"Error running local tests: {str(e)}"
    
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
            
            # Set up vector store with workspace files for file search
            if not self._setup_vector_store():
                logger.warning("Vector store setup failed, using code interpreter only")
            
            # Pre-load problem context to reduce API calls
            problem_context = ""
            try:
                # Read problem description
                problem_file = self.workspace_path / "problem.md"
                if problem_file.exists():
                    with open(problem_file, 'r', encoding='utf-8') as f:
                        problem_content = f.read()
                    problem_context += f"\n\nPROBLEM DESCRIPTION:\n{problem_content}"
                    
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

"""
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
        
        # Set total time limit for reasoning models (10 minutes)
        start_time = time.time()
        max_total_time = 600 if is_reasoning_model else 300  # 10 minutes for reasoning, 5 for others
        
        if is_reasoning_model:
            logger.info(f"Reasoning model {model} detected, allowing up to {max_iterations} iterations and {max_total_time//60} minutes total")
        
        while iteration < max_iterations and self.is_active:
            # Check total time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > max_total_time:
                logger.warning(f"Total time limit of {max_total_time//60} minutes exceeded ({elapsed_time:.1f}s)")
                break
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
                    logger.info(f"Using reasoning model {model} - allowing up to 10 minutes per request")
                
                while retry_count < max_retries and response is None:
                    try:
                        # Apply rate limiting before API call
                        self.rate_limiter.wait_if_needed_sync()
                        
                        # Build request for Responses API
                        request_params = {
                            "model": model,
                            "tools": self.tools,
                            "store": True,  # Store responses for continuity
                            "timeout": 600 if is_reasoning_model else 30
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
                
                # Download any output files from the response
                files_downloaded = self._download_output_files(response)
                if files_downloaded:
                    logger.info("Downloaded output files from OpenAI response")
                    
                    # Run local tests to see if solution works
                    tests_passed, test_output = self._run_local_tests()
                    logger.info(f"Local tests result: {'PASSED' if tests_passed else 'FAILED'}")
                    
                    if tests_passed:
                        logger.info("Solution passes all tests - completing successfully")
                        break
                    else:
                        logger.info("Tests failed - will provide feedback to continue conversation")
                        # Set up feedback for next iteration
                        self._pending_user_message = f"""The solution you created has some issues. Here are the test results:

{test_output}

Please analyze the test failures and create a corrected solution. Make sure to:
1. Fix any syntax or runtime errors
2. Handle all edge cases properly  
3. Ensure the solution follows the exact input/output format required

Please create an updated solution.py file."""
                
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