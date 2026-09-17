"""
LLM Provider Module
====================
Provider-agnostic interface for LLM operations supporting both Cloud (OpenAI)
and Local (Ollama) backends.

USAGE:
------
    # Cloud mode (default, GPT‑5)
    python main.py
    python main.py --cloud --model gpt-5
    
    # Local mode (Ollama)
    python main.py --local
    python main.py --local --model llava
    
    # Interactive selection
    python main.py --interactive
    
    # Environment variable auto-select
    export LLM_PROVIDER=local
    python main.py

ENVIRONMENT VARIABLES:
----------------------
    LLM_PROVIDER      - "cloud" or "local" (auto-selects, skips interactive prompt)
    LOCAL_LLM_MODEL   - Override default local model (e.g., "llama3.2")
    OLLAMA_HOST       - Override Ollama API URL (default: http://localhost:11434)
    OPENAI_API_KEY    - Required for cloud mode
"""

from .base import LLMProvider
from .factory import (
    auto_select_provider,
    create_provider,
    get_provider,
    get_provider_from_env,
    select_provider_gui,
    select_provider_interactive,
    select_provider_with_fallback,
)
from .gui_selector import select_llm_provider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .types import (
    LLMConfig,
    LLMError,
    LLMErrorType,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ProviderType,
)

__all__ = [
    "LLMConfig",
    "LLMError",
    "LLMErrorType",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMToolCall",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderType",
    "auto_select_provider",
    "create_provider",
    "get_provider",
    "get_provider_from_env",
    "select_llm_provider",
    "select_provider_gui",
    "select_provider_interactive",
    "select_provider_with_fallback",
]
