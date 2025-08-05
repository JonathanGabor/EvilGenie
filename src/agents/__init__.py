"""
Agent runners package.

This package contains all agent runner implementations organized into separate modules.
"""

from .base import AgentRunner
from .rate_limiter import OpenAIRateLimiter
from .claude_code import ClaudeCodeRunner
from .gemini_cli import GeminiCLIRunner
from .codex_cli import CodexCLIRunner

# Import optional agents that may not be available
try:
    from .smolagents import SmolagentsRunner
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False

try:
    from .openai_responses import OpenAIResponsesRunner
    OPENAI_RESPONSES_AVAILABLE = True
except ImportError as e:
    OPENAI_RESPONSES_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Failed to import OpenAIResponsesRunner: {e}")

# Registry of available agent runners
AGENT_RUNNERS = {
    "claude": ClaudeCodeRunner,
    "gemini": GeminiCLIRunner,
    "codex": CodexCLIRunner,
}

# Add optional agents if available
if SMOLAGENTS_AVAILABLE:
    AGENT_RUNNERS["smolagents"] = SmolagentsRunner

if OPENAI_RESPONSES_AVAILABLE:
    AGENT_RUNNERS["openai"] = OpenAIResponsesRunner

__all__ = [
    'AgentRunner',
    'OpenAIRateLimiter',
    'ClaudeCodeRunner', 
    'GeminiCLIRunner',
    'CodexCLIRunner',
    'AGENT_RUNNERS',
]

# Add optional exports
if SMOLAGENTS_AVAILABLE:
    __all__.append('SmolagentsRunner')

if OPENAI_RESPONSES_AVAILABLE:
    __all__.append('OpenAIResponsesRunner')