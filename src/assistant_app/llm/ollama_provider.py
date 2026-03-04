"""
Ollama Local Provider
=====================
LLM provider implementation for Ollama (local DeepSeek, LLaMA, etc.).

SUPPORTED MODELS:
-----------------
- llama3.2, llama3.1, llama2 - Meta's LLaMA family
- deepseek-r1:1.5b, deepseek-r1:7b - DeepSeek reasoning models
- mistral, mixtral - Mistral AI models
- llava, llava-llama3 - Vision-capable models (for screen analysis)
- qwen2.5 - Alibaba's Qwen models
- codellama - Code-focused models

ENVIRONMENT VARIABLES:
----------------------
- LOCAL_LLM_MODEL: Override default model (e.g., "llama3.2", "deepseek-r1:7b")
- OLLAMA_HOST: Override Ollama API URL (default: http://localhost:11434)

SETUP:
------
1. Install Ollama: brew install ollama (macOS) or see https://ollama.ai
2. Start Ollama: ollama serve
3. Pull a model: ollama pull llama3.2
4. Run assistant: python main.py --local

TOOL CALLING LIMITATIONS:
-------------------------
Ollama's tool/function calling support is experimental and model-dependent.
For full tool calling parity with OpenAI, use models like:
- mistral (best tool support)
- llama3.2 (limited support)

For models without native tool support, the assistant will:
1. Include tool descriptions in the system prompt
2. Parse tool calls from the model's text output
3. This may be less reliable than OpenAI's native function calling

VISION SUPPORT:
---------------
Only certain models support vision (image input):
- llava, llava-llama3, llava:7b, llava:13b
- bakllava
- moondream

For non-vision models, the assistant will fall back to text-only mode
with a note explaining the limitation.
"""

import os
import json
from typing import Any

import httpx

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

# Environment variable names
ENV_LOCAL_MODEL = "LOCAL_LLM_MODEL"
ENV_OLLAMA_HOST = "OLLAMA_HOST"


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM operations.
    
    Supports:
    - DeepSeek, LLaMA, Mistral, and other Ollama models
    - Vision with LLaVA and similar models
    - Basic tool calling (model-dependent)
    
    Ollama API Reference:
    - POST /api/chat - Chat completions
    - GET /api/tags - List available models
    - POST /api/generate - Text generation
    
    Environment Variables:
    - LOCAL_LLM_MODEL: Override model name
    - OLLAMA_HOST: Override API URL
    """
    
    # Models that support vision (multimodal)
    VISION_MODELS = {"llava", "llava-llama3", "bakllava", "moondream", "llava:7b", "llava:13b"}
    
    # Models that have good native tool calling support
    NATIVE_TOOL_MODELS = {"mistral", "mixtral", "llama3.2", "llama3.1", "qwen2.5"}
    
    # Models that can use tool-augmented prompting (JSON output)
    TOOL_MODELS = {"mistral", "mixtral", "deepseek-coder", "codellama", "qwen2.5", "llama3.2", "llama3.1", "deepseek-r1"}
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._available_models: list[str] | None = None
        
        # Override with environment variables if set
        env_model = os.environ.get(ENV_LOCAL_MODEL)
        if env_model:
            self.config.ollama_model = env_model
            logger.info(f"📝 Using model from {ENV_LOCAL_MODEL}: {env_model}")
        
        env_host = os.environ.get(ENV_OLLAMA_HOST)
        if env_host:
            self.config.ollama_base_url = env_host
            logger.info(f"📝 Using Ollama host from {ENV_OLLAMA_HOST}: {env_host}")
    
    @property
    def base_url(self) -> str:
        return self.config.ollama_base_url.rstrip("/")
    
    @property
    def name(self) -> str:
        return "Ollama"
    
    @property
    def model_name(self) -> str:
        return self.config.ollama_model
    
    @property
    def supports_vision(self) -> bool:
        if not self.config.enable_vision:
            return False
        model_base = self.config.ollama_model.split(":")[0].lower()
        return any(vm in model_base for vm in self.VISION_MODELS)
    
    @property
    def supports_tools(self) -> bool:
        """Whether this model can handle tools (native or augmented)."""
        if not self.config.enable_tools:
            return False
        model_base = self.config.ollama_model.split(":")[0].lower()
        return any(tm in model_base for tm in self.TOOL_MODELS)
    
    @property
    def supports_native_tools(self) -> bool:
        """Whether this model supports native Ollama tool calling."""
        if not self.config.enable_tools:
            return False
        model_base = self.config.ollama_model.split(":")[0].lower()
        return any(tm in model_base for tm in self.NATIVE_TOOL_MODELS)
    
    def is_available(self) -> tuple[bool, str]:
        """Check if Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    return False, f"Ollama returned status {response.status_code}"
                
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                self._available_models = models
                
                # Check if requested model is available
                model_name = self.config.ollama_model
                
                # Match exact or base name (e.g., "deepseek-r1:7b" matches "deepseek-r1:7b")
                model_found = any(
                    model_name == m or 
                    model_name.split(":")[0] == m.split(":")[0]
                    for m in models
                )
                
                if not model_found:
                    available_list = ", ".join(models[:5])
                    if len(models) > 5:
                        available_list += f"... ({len(models)} total)"
                    return False, (
                        f"Model '{model_name}' not found in Ollama. "
                        f"Available: {available_list}. "
                        f"Run: ollama pull {model_name}"
                    )
                
                return True, f"Connected to Ollama ({model_name})"
                
        except httpx.ConnectError:
            return False, (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except httpx.TimeoutException:
            return False, f"Timeout connecting to Ollama at {self.base_url}"
        except Exception as e:
            return False, f"Ollama error: {e}"
    
    def get_available_models(self) -> list[str]:
        """Get list of available models in Ollama."""
        if self._available_models is None:
            self.is_available()
        return self._available_models or []
    
    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a response using Ollama API."""
        try:
            # Prepare messages - may need to augment with tool descriptions
            processed_messages = messages.copy()
            use_native_tools = False
            
            if tools and self.supports_tools:
                if self.supports_native_tools:
                    # Use native Ollama tool calling
                    use_native_tools = True
                else:
                    # Use tool-augmented prompting for models without native support
                    processed_messages = self._augment_messages_with_tools(messages, tools)
                    logger.debug("🔧 Using tool-augmented prompting (no native tool support)")
            
            # Build request payload
            payload: dict[str, Any] = {
                "model": self.config.ollama_model,
                "messages": [self._convert_message(m) for m in processed_messages],
                "stream": False,
                "options": {
                    "num_predict": max_tokens or self.config.max_tokens,
                    "temperature": temperature if temperature is not None else self.config.temperature,
                },
            }
            
            # Add tools if using native tool calling
            if use_native_tools and tools:
                payload["tools"] = self._convert_tools(tools)
            
            logger.debug(f"🏠 Ollama request: model={self.config.ollama_model}, messages={len(messages)}")
            
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    raise self._handle_error(response.status_code, error_text)
                
                return self._parse_response(response.json())
                
        except httpx.ConnectError:
            raise LLMError(
                error_type=LLMErrorType.PROVIDER_UNAVAILABLE,
                message=f"Cannot connect to Ollama at {self.base_url}",
                user_message="Ollama is not running. Please start it with: ollama serve",
                recoverable=False,
                provider=self.name,
            )
        except httpx.TimeoutException:
            raise LLMError(
                error_type=LLMErrorType.TIMEOUT,
                message=f"Ollama request timed out after {self.config.timeout}s",
                user_message="Request to local model timed out. The model might be loading.",
                recoverable=True,
                provider=self.name,
            )
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                error_type=LLMErrorType.UNKNOWN,
                message=str(e),
                user_message="An error occurred with the local model.",
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
            # For non-vision models, provide a helpful message
            logger.warning(
                f"Model {self.config.ollama_model} doesn't support vision. "
                "Using text-only mode. For vision, use: llava, bakllava, or moondream"
            )
            
            # Fall back to text-only with a note
            messages: list[LLMMessage] = []
            
            if system_prompt:
                messages.append(LLMMessage(role="system", content=system_prompt))
            
            if history:
                messages.extend(history)
            
            # Add context about the limitation
            fallback_prompt = (
                f"{prompt}\n\n"
                "[Note: I cannot see your screen with this model. "
                "Please describe what you see or switch to a vision model like 'llava'.]"
            )
            messages.append(LLMMessage(role="user", content=fallback_prompt))
            
            return self.generate(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        # Build messages with vision content
        messages = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        if history:
            messages.extend(history)
        
        # Create vision message - Ollama uses "images" array
        messages.append(LLMMessage(
            role="user",
            content=prompt,
        ))
        
        # For Ollama, we need to pass images separately
        return self._generate_with_images(
            messages=messages,
            images=[image_data.decode()],  # Base64 string
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def _generate_with_images(
        self,
        messages: list[LLMMessage],
        images: list[str],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate with images using Ollama's format."""
        try:
            # Build request payload with images
            ollama_messages = []
            for m in messages:
                msg = self._convert_message(m)
                ollama_messages.append(msg)
            
            # Add images to the last user message
            if ollama_messages and ollama_messages[-1]["role"] == "user":
                ollama_messages[-1]["images"] = images
            
            payload: dict[str, Any] = {
                "model": self.config.ollama_model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens or self.config.max_tokens,
                    "temperature": temperature if temperature is not None else self.config.temperature,
                },
            }
            
            logger.debug(f"🏠 Ollama vision request: model={self.config.ollama_model}")
            
            with httpx.Client(timeout=self.config.timeout * 2) as client:  # Longer timeout for vision
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                
                if response.status_code != 200:
                    raise self._handle_error(response.status_code, response.text)
                
                return self._parse_response(response.json())
                
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                error_type=LLMErrorType.UNKNOWN,
                message=str(e),
                user_message="Vision processing failed with local model.",
                recoverable=True,
                provider=self.name,
                original_error=e,
            )
    
    def _convert_message(self, message: LLMMessage) -> dict[str, Any]:
        """Convert LLMMessage to Ollama format."""
        msg: dict[str, Any] = {
            "role": message.role,
            "content": message.content if isinstance(message.content, str) else str(message.content),
        }
        return msg
    
    def _augment_messages_with_tools(
        self, 
        messages: list[LLMMessage], 
        tools: list[dict[str, Any]]
    ) -> list[LLMMessage]:
        """
        Augment messages with tool descriptions for models without native tool support.
        
        This implements a structured prompting approach where tools are described
        in the system prompt and the model is asked to respond with JSON when
        it wants to call a tool.
        """
        # Build tool descriptions
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "No description")
                params = func.get("parameters", {})
                
                param_desc = []
                properties = params.get("properties", {})
                required = params.get("required", [])
                for param_name, param_info in properties.items():
                    req = "(required)" if param_name in required else "(optional)"
                    param_desc.append(f"    - {param_name}: {param_info.get('description', 'No description')} {req}")
                
                tool_descriptions.append(f"""
**{name}**: {desc}
  Parameters:
{chr(10).join(param_desc) if param_desc else '    (none)'}
""")
        
        tool_prompt = f"""
You have access to the following tools. When you need to use a tool, respond with ONLY a JSON object in this exact format:
{{"tool_call": {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}}}

Available tools:
{''.join(tool_descriptions)}

IMPORTANT: 
- Only use a tool if the user's request clearly requires it
- If you don't need a tool, respond normally with text
- When using a tool, output ONLY the JSON, nothing else
"""
        
        # Prepend or append tool info to system message
        augmented = []
        system_found = False
        
        for msg in messages:
            if msg.role == "system" and not system_found:
                # Append tool info to existing system prompt
                new_content = f"{msg.content}\n\n{tool_prompt}"
                augmented.append(LLMMessage(role="system", content=new_content))
                system_found = True
            else:
                augmented.append(msg)
        
        # If no system message, add one
        if not system_found:
            augmented.insert(0, LLMMessage(role="system", content=tool_prompt))
        
        return augmented
    
    def _parse_augmented_tool_calls(self, content: str) -> list[LLMToolCall]:
        """
        Parse tool calls from model output when using augmented prompting.
        
        Looks for JSON in the format: {"tool_call": {"name": "...", "arguments": {...}}}
        """
        import re
        
        tool_calls = []
        
        # Try to find JSON tool call pattern
        try:
            # First try: direct JSON parse if the whole content is JSON
            if content.strip().startswith('{') and '"tool_call"' in content:
                try:
                    data = json.loads(content.strip())
                    if "tool_call" in data:
                        tc = data["tool_call"]
                        tool_calls.append(LLMToolCall(
                            id=f"augmented_call_0",
                            name=tc.get("name", ""),
                            arguments=json.dumps(tc.get("arguments", {})),
                        ))
                        return tool_calls
                except json.JSONDecodeError:
                    pass
            
            # Second try: find and extract JSON from mixed content
            # Find the start of a JSON object containing tool_call
            start_idx = content.find('{"tool_call"')
            if start_idx == -1:
                start_idx = content.find('{ "tool_call"')
            
            if start_idx >= 0:
                # Extract everything from the opening brace and try to parse
                remaining = content[start_idx:]
                # Count braces to find the end of the JSON object
                brace_count = 0
                end_idx = 0
                for i, char in enumerate(remaining):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > 0:
                    json_str = remaining[:end_idx]
                    try:
                        data = json.loads(json_str)
                        if "tool_call" in data:
                            tc = data["tool_call"]
                            tool_calls.append(LLMToolCall(
                                id=f"augmented_call_0",
                                name=tc.get("name", ""),
                                arguments=json.dumps(tc.get("arguments", {})),
                            ))
                    except json.JSONDecodeError:
                        pass
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.debug(f"Could not parse augmented tool call: {e}")
        
        return tool_calls
    
    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Ollama format."""
        # Ollama uses a similar format to OpenAI for tools
        ollama_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                ollama_tools.append({
                    "type": "function",
                    "function": tool.get("function", {}),
                })
        return ollama_tools
    
    def _parse_response(self, data: dict[str, Any], check_augmented_tools: bool = True) -> LLMResponse:
        """Parse Ollama response into LLMResponse."""
        message = data.get("message", {})
        content = message.get("content", "")
        
        # Extract tool calls if present (native Ollama tool calling)
        tool_calls: list[LLMToolCall] = []
        if "tool_calls" in message:
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                tool_calls.append(LLMToolCall(
                    id=f"call_{i}",
                    name=func.get("name", ""),
                    arguments=json.dumps(func.get("arguments", {})),
                ))
        
        # Check for augmented tool calls in content (for non-native tool models)
        if not tool_calls and check_augmented_tools and content:
            augmented_calls = self._parse_augmented_tool_calls(content)
            if augmented_calls:
                tool_calls = augmented_calls
                # Clear content if it was just a tool call
                if '{"tool_call"' in content:
                    content = ""
        
        # Determine finish reason
        finish_reason = "stop"
        if data.get("done_reason"):
            finish_reason = data["done_reason"]
        elif tool_calls:
            finish_reason = "tool_calls"
        
        return LLMResponse(
            content=content if content else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=data.get("model", self.config.ollama_model),
            provider=self.name,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            raw=data,
        )
    
    def _handle_error(self, status_code: int, error_text: str) -> LLMError:
        """Convert HTTP error to LLMError."""
        error_lower = error_text.lower()
        
        if status_code == 404 or "not found" in error_lower:
            return LLMError(
                error_type=LLMErrorType.MODEL_NOT_FOUND,
                message=error_text,
                user_message=f"Model '{self.config.ollama_model}' not found. Run: ollama pull {self.config.ollama_model}",
                recoverable=False,
                provider=self.name,
            )
        
        if status_code == 500 or "error" in error_lower:
            return LLMError(
                error_type=LLMErrorType.SERVER_ERROR,
                message=error_text,
                user_message="Ollama server error. The model may still be loading.",
                recoverable=True,
                provider=self.name,
            )
        
        return LLMError(
            error_type=LLMErrorType.UNKNOWN,
            message=f"HTTP {status_code}: {error_text}",
            user_message="An error occurred with the local model.",
            recoverable=True,
            provider=self.name,
        )
