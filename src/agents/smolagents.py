"""
Smolagents agent runner implementation.
"""

import os
import time
import logging
import threading
from typing import Dict, Any

from .base import AgentRunner
from env_utils import build_subprocess_env

logger = logging.getLogger(__name__)

# Try to import smolagents, fall back gracefully if not available
try:
    from smolagents import CodeAgent, OpenAIModel, LiteLLMModel, tool
    import subprocess
    import json
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
        
    def _create_model(self, provider: str, model_name: str):
        """Create a model instance based on the provider.
        
        Args:
            provider: Model provider ('openai', 'anthropic', 'litellm', 'auto')
            model_name: Model name/ID
            
        Returns:
            Model instance or None if failed
        """
        try:
            if provider.lower() == 'openai':
                # Use OpenAIModel for direct OpenAI integration
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    logger.error("OPENAI_API_KEY environment variable not set")
                    return None
                return OpenAIModel(model_id=model_name, api_key=api_key)
            
            elif provider.lower() == 'anthropic':
                # Use LiteLLMModel for Anthropic models
                api_key = os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.error("ANTHROPIC_API_KEY environment variable not set")
                    return None
                # Format model name for LiteLLM (anthropic/model-name)
                litellm_model = f"anthropic/{model_name}" if not model_name.startswith("anthropic/") else model_name
                return LiteLLMModel(model_id=litellm_model)
            
            elif provider.lower() in ['litellm', 'auto']:
                # LiteLLM can handle multiple providers via unified interface
                # Requires appropriate API keys set for the specific provider
                return LiteLLMModel(model_id=model_name)
            
            else:
                # Default to LiteLLM for unknown providers (it supports 100+ providers)
                logger.info(f"Using LiteLLM for provider: {provider}")
                return LiteLLMModel(model_id=model_name)
                
        except Exception as e:
            logger.error(f"Error creating {provider} model: {e}")
            return None
    
    def _create_programming_tools(self):
        """Create comprehensive programming tools for competitive programming."""
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
        
        @tool
        def run_tests() -> str:
            """Run the test.py script to validate the solution.
            
            Returns:
                Test results output
            """
            try:
                result = subprocess.run(
                    ['python', 'test.py'],
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=build_subprocess_env()  # No secrets needed for tests
                )
                
                output = f"Exit code: {result.returncode}\n"
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n"
                
                logger.info(f"Ran tests with exit code {result.returncode}")
                return output
                
            except subprocess.TimeoutExpired:
                error_msg = "Test execution timed out (30s limit)"
                logger.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"Error running tests: {e}"
                logger.error(error_msg)
                return error_msg
        
        @tool
        def validate_solution() -> str:
            """Validate that solution.py exists and has basic structure.
            
            Returns:
                Validation results
            """
            try:
                solution_path = workspace_path / 'solution.py'
                if not solution_path.exists():
                    return "ERROR: solution.py file does not exist"
                
                with open(solution_path, 'r') as f:
                    content = f.read()
                
                # Basic validation checks
                checks = {
                    'has_main_guard': '__name__' in content and '__main__' in content,
                    'has_function_def': 'def ' in content,
                    'has_input_handling': 'input()' in content,
                    'has_print_output': 'print(' in content,
                    'non_empty': len(content.strip()) > 0
                }
                
                passed = sum(checks.values())
                total = len(checks)
                
                result = f"Solution validation: {passed}/{total} checks passed\n"
                for check, status in checks.items():
                    result += f"  {check}: {'✓' if status else '✗'}\n"
                
                logger.info(f"Solution validation: {passed}/{total} checks passed")
                return result
                
            except Exception as e:
                error_msg = f"Error validating solution: {e}"
                logger.error(error_msg)
                return error_msg
        
        @tool
        def analyze_test_cases() -> str:
            """Analyze the test_cases.json file to understand input/output patterns.
            
            Returns:
                Analysis of test cases
            """
            try:
                test_cases_path = workspace_path / 'test_cases.json'
                if not test_cases_path.exists():
                    return "ERROR: test_cases.json file does not exist"
                
                with open(test_cases_path, 'r') as f:
                    test_cases = json.load(f)
                
                analysis = f"Found {len(test_cases)} test cases:\n\n"
                
                for i, test_case in enumerate(test_cases[:3]):  # Show first 3 test cases
                    analysis += f"Test Case {i+1}:\n"
                    analysis += f"  Input: {repr(test_case.get('input', 'N/A'))}\n"
                    analysis += f"  Output: {repr(test_case.get('output', 'N/A'))}\n\n"
                
                if len(test_cases) > 3:
                    analysis += f"... and {len(test_cases) - 3} more test cases\n"
                
                logger.info(f"Analyzed {len(test_cases)} test cases")
                return analysis
                
            except Exception as e:
                error_msg = f"Error analyzing test cases: {e}"
                logger.error(error_msg)
                return error_msg
        
        return [read_file, write_file, list_files, run_tests, validate_solution, analyze_test_cases]
    
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
            
            # Create model based on provider configuration
            model_config = self.config.get('model_config', {})
            provider = model_config.get('provider', 'openai')
            model_name = model_config.get('model', 'gpt-4o-mini')
            
            model = self._create_model(provider, model_name)
            if not model:
                return False
            
            # Create comprehensive programming tools
            programming_tools = self._create_programming_tools()
            
            # Create agent with enhanced configuration
            self.agent = CodeAgent(
                tools=programming_tools,
                model=model,
                additional_authorized_imports=[
                    'sys', 'json', 'math', 'itertools', 'collections', 
                    'heapq', 'bisect', 'functools', 'operator', 're',
                    'decimal', 'fractions', 'string', 'random'
                ],
                verbosity_level=1   # Moderate verbosity for debugging
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
2. Analyze the test cases using analyze_test_cases() to understand input/output format
3. Write a COMPLETE new solution.py file with your implementation
4. Validate your solution using validate_solution()
5. Run tests using run_tests() to verify correctness
6. If tests fail, analyze the output and fix your solution
7. The solution must be a complete Python program with proper input/output handling

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
