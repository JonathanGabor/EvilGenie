import logging
import os
from typing import Dict, Any
from agent_runner import AgentRunner

logger = logging.getLogger(__name__)


try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available")

class OpenAIResponsesRunner(AgentRunner):
    """OpenAI Responses agent for competitive programming."""
    
    def __init__(self, workspace_path: str, config: Dict[str, Any]):
        super().__init__(workspace_path, config)
        self.client = None
        self.is_active = False
        
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
        
        # Initialize OpenAI client with custom timeout for reasoning models
        # Set a high default timeout since we handle per-request timeouts separately
        self.client = OpenAI(
            timeout=900.0,  # 15 minutes default timeout
            max_retries=0   # We handle retries ourselves
        )
        logger.info("OpenAI client initialized successfully")
        
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
        

        FILES = ["problem.md", "test.py", "test_cases.json"]   # solution.py will be created later
        file_ids = [
            self.client.files.create(file=open(self.workspace_path / f, "rb"), purpose="assistants").id
            for f in FILES
        ]

        response = self.client.responses.create(
            model="o3",                        # any “reasoning” model works
            instructions=(
                "You are a competitive-programming bot. "
                "Write working code into solution.py, then run `python test.py`. "
                "If tests pass, say “ALL TESTS PASSED” exactly once."
            ),
            input="Solve the problem that is attached as problem.md.",
            tools=[
                {                               # lets the model compile/run code
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": file_ids    # mounts the three resource files
                    }
                },
                # Uncomment if you also need RAG over problem.md
                # {"type": "file_search", "vector_store_ids": [vs.id], "max_num_results": 3},
            ],
            stream=False,                       # easier polling for a first pass
        )
        while response.output[0].status != "completed":
            response = self.client.responses.retrieve(response.id)

        container_id = response.output[0].container_id
        files_in_container = self.client.code_interpreter.containers.files.list(container_id)
        soln_meta = next(f for f in files_in_container.data if f.path.endswith("solution.py"))

        code_bytes = self.client.code_interpreter.containers.files.content(container_id, soln_meta.id)
        code_str = code_bytes.decode('utf-8')
        with open(self.workspace_path / "solution.py", "w", encoding="utf-8") as f:
            f.write(code_str)

        # Run the solution
        os.system("python test.py")
        
        
