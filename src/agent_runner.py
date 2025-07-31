"""
Agent runner module for launching and controlling coding agents.

Supports various coding agents like Claude Code, Gemini CLI, etc.
Each agent is launched in a subprocess with a specific workspace directory.
"""

import logging
from typing import Dict, Optional, Any

try:
    # Try relative import first (when used as module)
    from .agents import (
        AgentRunner,
        OpenAIRateLimiter,
        ClaudeCodeRunner,
        GeminiCLIRunner,
        CodexCLIRunner,
        AGENT_RUNNERS,
    )
    
    # Import optional agents
    try:
        from .agents import SmolagentsRunner
    except ImportError:
        pass

    try:
        from .agents import OpenAIFunctionCallingRunner
    except ImportError:
        pass
        
except ImportError:
    # Fall back to absolute import (when run directly)
    from agents import (
        AgentRunner,
        OpenAIRateLimiter,
        ClaudeCodeRunner,
        GeminiCLIRunner,
        CodexCLIRunner,
        AGENT_RUNNERS,
    )
    
    # Import optional agents
    try:
        from agents import SmolagentsRunner
    except ImportError:
        pass

    try:
        from agents import OpenAIFunctionCallingRunner
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    'AgentRunner',
    'OpenAIRateLimiter', 
    'ClaudeCodeRunner',
    'GeminiCLIRunner',
    'CodexCLIRunner',
    'AGENT_RUNNERS',
    'create_agent_runner',
]

# Add optional exports if available (check if already imported)
if 'SmolagentsRunner' in globals():
    __all__.append('SmolagentsRunner')

if 'OpenAIFunctionCallingRunner' in globals():
    __all__.append('OpenAIFunctionCallingRunner')


def create_agent_runner(agent_type: str, workspace_path: str, config: Dict[str, Any]) -> Optional[AgentRunner]:
    """Factory function to create an agent runner."""
    runner_class = AGENT_RUNNERS.get(agent_type.lower())
    if not runner_class:
        logger.error(f"Unknown agent type: {agent_type}")
        return None
    
    return runner_class(workspace_path, config)