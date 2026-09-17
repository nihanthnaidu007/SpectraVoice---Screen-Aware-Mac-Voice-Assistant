"""
LLM Provider Types
==================
Shared types and data structures for LLM providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderType(Enum):
    """Supported LLM provider types."""
    CLOUD = "cloud"    # OpenAI GPT
    LOCAL = "local"    # Ollama (DeepSeek, LLaMA, etc.)


class LLMErrorType(Enum):
    """Types of LLM errors for appropriate handling."""
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_API_KEY = "invalid_api_key"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_TOO_LONG = "context_too_long"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    INVALID_REQUEST = "invalid_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNKNOWN = "unknown"


@dataclass
class LLMError(Exception):
    """Structured LLM error with type classification."""
    error_type: LLMErrorType
    message: str
    user_message: str
    recoverable: bool = True
    provider: str = ""
    original_error: Exception | None = None
    
    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: ProviderType = ProviderType.CLOUD
    
    # Model settings (Cloud default is GPT‑5)
    model: str = "gpt-5"
    max_tokens: int = 800
    temperature: float = 0.7
    
    # Ollama-specific settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    
    # Timeout settings
    timeout: int = 60
    
    # Feature flags
    enable_vision: bool = True
    enable_tools: bool = True
    
    @classmethod
    def for_cloud(cls, model: str = "gpt-5", **kwargs) -> "LLMConfig":
        """Create config for cloud (OpenAI GPT‑5) provider."""
        return cls(provider=ProviderType.CLOUD, model=model, **kwargs)
    
    @classmethod
    def for_local(
        cls, 
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> "LLMConfig":
        """Create config for local (Ollama) provider."""
        return cls(
            provider=ProviderType.LOCAL,
            ollama_model=model,
            ollama_base_url=base_url,
            **kwargs
        )


@dataclass
class LLMMessage:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Text or multimodal content
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["LLMToolCall"] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API calls."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class LLMToolCall:
    """A tool/function call from the LLM."""
    id: str
    name: str
    arguments: str  # JSON string
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            }
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""
    
    # Usage statistics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Raw response for debugging
    raw: Any = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0
    
    @property
    def is_complete(self) -> bool:
        """Check if response completed normally."""
        return self.finish_reason in ("stop", "end_turn")
    
    @property
    def is_tool_use(self) -> bool:
        """Check if response is requesting tool use."""
        return self.finish_reason == "tool_calls" or self.has_tool_calls


@dataclass
class GenerateRequest:
    """Request parameters for LLM generation."""
    messages: list[LLMMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        params: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages]
        }
        if self.tools:
            params["tools"] = self.tools
        if self.tool_choice:
            params["tool_choice"] = self.tool_choice
        if self.max_tokens:
            params["max_completion_tokens"] = self.max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params
