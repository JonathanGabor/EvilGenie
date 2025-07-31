"""
Smolagents agent runner implementation.
"""

import os
import time
import logging
import threading
from typing import Dict, Any

from .base import AgentRunner

logger = logging.getLogger(__name__)

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