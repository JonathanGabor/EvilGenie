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
    from .openai_function import OpenAIFunctionCallingRunner
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Registry of available agent runners
AGENT_RUNNERS = {
    "claude": ClaudeCodeRunner,
    "gemini": GeminiCLIRunner,
    "codex": CodexCLIRunner,
}

# Add optional agents if available
if SMOLAGENTS_AVAILABLE:
    AGENT_RUNNERS["smolagents"] = SmolagentsRunner

if OPENAI_AVAILABLE:
    AGENT_RUNNERS["openai"] = OpenAIFunctionCallingRunner

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

if OPENAI_AVAILABLE:
    __all__.append('OpenAIFunctionCallingRunner')