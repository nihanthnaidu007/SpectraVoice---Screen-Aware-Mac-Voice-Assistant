"""
OpenAI Cloud Provider
=====================
LLM provider implementation for OpenAI GPT models (cloud).
"""

import os
from typing import Any

import openai

from assistant_app.utils.logging_config import get_logger
from .base import LLMProvider
from .types import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMError,
    LLMErrorType,
)

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider for cloud-based LLM operations.
    
    Supports:
    - Vision (GPT-5 family)
    - Function/tool calling
    - Conversation context
    """
    
    # Models that support vision (GPT‑5+ only)
    VISION_MODELS = {"gpt-5", "gpt-5.1", "gpt-5.2"}
    
    # Models that support function calling (GPT‑5+ only)
    TOOL_MODELS = {"gpt-5", "gpt-5.1", "gpt-5.2"}
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: openai.OpenAI | None = None
    
    @property
    def client(self) -> openai.OpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    @property
    def model_name(self) -> str:
        return self.config.model
    
    @property
    def supports_vision(self) -> bool:
        return self.config.enable_vision and self.config.model in self.VISION_MODELS
    
    @property
    def supports_tools(self) -> bool:
        return self.config.enable_tools and self.config.model in self.TOOL_MODELS
    
    def is_available(self) -> tuple[bool, str]:
        """Check if OpenAI API is accessible."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY environment variable not set"
        
        if not api_key.startswith(("sk-", "sess-")):
            return False, "Invalid OpenAI API key format"
        
        try:
            # Quick validation with models endpoint
            self.client.models.list()
            return True, f"Connected to OpenAI ({self.model_name})"
        except openai.AuthenticationError:
            return False, "Invalid OpenAI API key"
        except openai.APIConnectionError as e:
            return False, f"Cannot connect to OpenAI API: {e}"
        except Exception as e:
            return False, f"OpenAI API error: {e}"
    
    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a response using OpenAI API."""
        try:
            # Build request parameters
            params: dict[str, Any] = {
                "model": self.config.model,
                "messages": [self._convert_message(m) for m in messages],
                "max_completion_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature if temperature is not None else self.config.temperature,
            }
            
            # Add tools if supported and provided
            if tools and self.supports_tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice or "auto"
            
            logger.debug(f"🌐 OpenAI request: model={self.config.model}, messages={len(messages)}")
            
            response = self.client.chat.completions.create(**params)
            
            return self._parse_response(response)
            
        except openai.RateLimitError as e:
            raise LLMError(
                error_type=LLMErrorType.RATE_LIMITED,
                message=str(e),
                user_message="API rate limit reached. Please wait a moment.",
                recoverable=True,
                provider=self.name,
                original_error=e,
            )
        except openai.AuthenticationError as e:
            raise LLMError(
                error_type=LLMErrorType.INVALID_API_KEY,
                message=str(e),
                user_message="Invalid OpenAI API key. Please check your configuration.",
                recoverable=False,
                provider=self.name,
                original_error=e,
            )
        except openai.APIConnectionError as e:
            raise LLMError(
                error_type=LLMErrorType.CONNECTION_ERROR,
                message=str(e),
                user_message="Cannot connect to OpenAI. Check your internet connection.",
                recoverable=True,
                provider=self.name,
                original_error=e,
            )
        except openai.BadRequestError as e:
            error_msg = str(e).lower()
            if "context_length" in error_msg or "maximum context" in error_msg:
                raise LLMError(
                    error_type=LLMErrorType.CONTEXT_TOO_LONG,
                    message=str(e),
                    user_message="Message too long. Please shorten your request.",
                    recoverable=True,
                    provider=self.name,
                    original_error=e,
                )
            raise LLMError(
                error_type=LLMErrorType.INVALID_REQUEST,
                message=str(e),
                user_message="Invalid request to OpenAI.",
                recoverable=False,
                provider=self.name,
                original_error=e,
            )
        except openai.APIStatusError as e:
            raise LLMError(
                error_type=LLMErrorType.SERVER_ERROR,
                message=str(e),
                user_message="OpenAI server error. Please try again.",
                recoverable=True,
                provider=self.name,
                original_error=e,
            )
        except Exception as e:
            raise LLMError(
                error_type=LLMErrorType.UNKNOWN,
                message=str(e),
                user_message="An unexpected error occurred with OpenAI.",
                recoverable=True,
                provider=self.name,
                original_error=e,
            )
    
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
        """Generate a response with vision input."""
        if not self.supports_vision:
            raise LLMError(
                error_type=LLMErrorType.INVALID_REQUEST,
                message=f"Model {self.config.model} does not support vision",
                user_message="Current model doesn't support image analysis.",
                recoverable=False,
                provider=self.name,
            )
        
        # Build messages with vision content
        messages: list[LLMMessage] = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        if history:
            messages.extend(history)
        
        # Create vision message with image
        image_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data.decode()}"}
            }
        ]
        messages.append(LLMMessage(role="user", content=image_content))
        
        return self.generate(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def _convert_message(self, message: LLMMessage) -> dict[str, Any]:
        """Convert LLMMessage to OpenAI format."""
        msg: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        
        if message.name:
            msg["name"] = message.name
        
        if message.tool_call_id:
            msg["tool_call_id"] = message.tool_call_id
        
        if message.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in message.tool_calls]
        
        return msg
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse."""
        if not response.choices:
            return LLMResponse(
                content=None,
                finish_reason="empty",
                model=response.model,
                provider=self.name,
            )
        
        choice = response.choices[0]
        message = choice.message
        
        # Extract tool calls if present
        tool_calls: list[LLMToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(LLMToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))
        
        # Extract usage
        usage = response.usage or {}
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            model=response.model,
            provider=self.name,
            prompt_tokens=getattr(usage, 'prompt_tokens', 0),
            completion_tokens=getattr(usage, 'completion_tokens', 0),
            total_tokens=getattr(usage, 'total_tokens', 0),
            raw=response,
        )
