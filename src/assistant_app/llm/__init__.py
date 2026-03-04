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

from .types import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMError,
    LLMErrorType,
    ProviderType,
)
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .factory import (
    create_provider, 
    get_provider,
    get_provider_from_env,
    auto_select_provider,
    select_provider_interactive,
    select_provider_gui,
    select_provider_with_fallback,
)
from .gui_selector import select_llm_provider

__all__ = [
    # Types
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMToolCall",
    "LLMError",
    "LLMErrorType",
    "ProviderType",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "OllamaProvider",
    # Factory
    "create_provider",
    "get_provider",
    "get_provider_from_env",
    "auto_select_provider",
    "select_provider_interactive",
    "select_provider_gui",
    "select_provider_with_fallback",
    # GUI Selector
    "select_llm_provider",
]
