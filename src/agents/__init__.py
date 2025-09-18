"""
Agent runners package.

This package contains all agent runner implementations organized into separate modules.
"""

import logging
from typing import Dict, Optional, Any

from .base import AgentRunner
from .claude_code import ClaudeCodeRunner
from .gemini_cli import GeminiCLIRunner
from .codex_cli import CodexCLIRunner
from .smolagents import SmolagentsRunner

try:
    from .openai_responses import OpenAIResponsesRunner
    OPENAI_RESPONSES_AVAILABLE = True
except ImportError as e:
    OPENAI_RESPONSES_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Failed to import OpenAIResponsesRunner: {e}")

try:
    from .openhands_cli import OpenHandsCLIRunner
    OPENHANDS_AVAILABLE = True
except ImportError as e:
    OPENHANDS_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Failed to import OpenHandsCLIRunner: {e}")

# Registry of available agent runners
AGENT_RUNNERS = {
    "claude": ClaudeCodeRunner,
    "gemini": GeminiCLIRunner,
    "codex": CodexCLIRunner,
    "smolagents": SmolagentsRunner,
}

# Add optional agents if available

if OPENAI_RESPONSES_AVAILABLE:
    AGENT_RUNNERS["openai"] = OpenAIResponsesRunner

if OPENHANDS_AVAILABLE:
    AGENT_RUNNERS["openhands"] = OpenHandsCLIRunner

__all__ = [
    'AgentRunner',
    'ClaudeCodeRunner', 
    'GeminiCLIRunner',
    'CodexCLIRunner',
    'AGENT_RUNNERS',
    'SmolagentsRunner',
    'create_agent_runner',
]

# Add optional exports
if OPENAI_RESPONSES_AVAILABLE:
    __all__.append('OpenAIResponsesRunner')

if OPENHANDS_AVAILABLE:
    __all__.append('OpenHandsCLIRunner')

logger = logging.getLogger(__name__)


def create_agent_runner(agent_type: str, workspace_path: str, config: Dict[str, Any]) -> Optional[AgentRunner]:
    """Factory function to create an agent runner."""
    runner_class = AGENT_RUNNERS.get(agent_type.lower())
    if not runner_class:
        logger.error(f"Unknown agent type: {agent_type}")
        logger.error(f"Available agents: {list(AGENT_RUNNERS.keys())}")
        return None
    
    return runner_class(workspace_path, config)
