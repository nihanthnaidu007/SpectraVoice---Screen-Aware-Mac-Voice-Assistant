"""
LLM Provider Base Class
=======================
Abstract base class defining the provider-agnostic interface for LLM operations.

PROVIDER-AGNOSTIC DESIGN
------------------------
This module implements the Strategy Pattern for LLM providers:

1. **LLMProvider** (this file): Abstract interface that all providers implement
2. **OpenAIProvider** (openai_provider.py): Cloud implementation using OpenAI GPT
3. **OllamaProvider** (ollama_provider.py): Local implementation using Ollama

The calling code (Assistant class) only interacts with the LLMProvider interface,
allowing seamless switching between Cloud and Local modes without code changes.

USAGE EXAMPLE
-------------
```python
from assistant_app.llm import create_provider, LLMConfig

# Cloud mode (GPT‑5)
config = LLMConfig.for_cloud(model="gpt-5")
provider = create_provider(config)

# Local mode
config = LLMConfig.for_local(model="llama3.2")
provider = create_provider(config)

# Use identically regardless of provider
response = provider.generate([LLMMessage(role="user", content="Hello")])
```

KEY INTERFACE METHODS
---------------------
- `generate()`: Text-only generation with optional tool calling
- `generate_with_vision()`: Generation with image input (screen analysis)
- `is_available()`: Check if provider is ready (API key valid, Ollama running)
- `supports_vision`: Property indicating vision capability
- `supports_tools`: Property indicating tool/function calling capability
"""

from abc import ABC, abstractmethod
from typing import Any

from .types import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMError,
)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers (Cloud/OpenAI, Local/Ollama) must implement this interface.
    The calling code should not care which provider is being used.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the provider."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Currently configured model name."""
        pass
    
    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider/model supports vision (image) input."""
        pass
    
    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider/model supports function/tool calling."""
        pass
    
    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """
        Check if the provider is available and ready.
        
        Returns:
            Tuple of (is_available, status_message)
            - For cloud: checks API key validity
            - For local: checks if Ollama is running and model exists
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Conversation history as list of messages
            tools: Optional list of tool schemas (function calling)
            tool_choice: Optional tool choice directive ("auto", "none", or specific)
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            LLMResponse with content and/or tool calls
            
        Raises:
            LLMError: On any provider-specific error
        """
        pass
    
    @abstractmethod
    def generate_with_vision(
        self,
        prompt: str,
        image_data: bytes,
        system_prompt: str | None = None,
        history: list[LLMMessage] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Generate a response with vision (image) input.
        
        Args:
            prompt: User text prompt
            image_data: Base64-encoded image data
            system_prompt: Optional system prompt
            history: Optional conversation history
            tools: Optional tool schemas
            tool_choice: Optional tool choice directive
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            LLMResponse with content and/or tool calls
            
        Raises:
            LLMError: On any provider-specific error, including lack of vision support
        """
        pass
    
    def get_status(self) -> dict[str, Any]:
        """
        Get provider status information.
        
        Returns:
            Dictionary with provider status details
        """
        available, message = self.is_available()
        return {
            "provider": self.name,
            "model": self.model_name,
            "available": available,
            "message": message,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
        }
