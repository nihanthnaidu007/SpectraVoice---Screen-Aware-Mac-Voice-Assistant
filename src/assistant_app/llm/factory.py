"""
LLM Provider Factory
====================
Factory for creating and managing LLM providers based on configuration.

ENVIRONMENT VARIABLES:
----------------------
- LLM_PROVIDER: Set to "cloud" or "local" to auto-select provider without prompting
- LOCAL_LLM_MODEL: Override default local model name
- OLLAMA_HOST: Override Ollama API URL
- OPENAI_API_KEY: Required for cloud mode

GUI SELECTION:
--------------
By default, running `python main.py` will show a GUI pop-up for selecting the provider.
Set LLM_PROVIDER environment variable to skip the GUI in non-interactive environments.
"""

import os
from typing import Any

from assistant_app.utils.logging_config import get_logger
from .base import LLMProvider
from .types import LLMConfig, ProviderType, LLMError, LLMErrorType
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

logger = get_logger(__name__)

# Environment variable names
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LOCAL_MODEL = "LOCAL_LLM_MODEL"
ENV_OLLAMA_HOST = "OLLAMA_HOST"

# Global provider instance (singleton pattern)
_provider: LLMProvider | None = None
_current_config: LLMConfig | None = None


def create_provider(config: LLMConfig) -> LLMProvider:
    """
    Create an LLM provider based on configuration.
    
    Args:
        config: LLM configuration specifying provider type and settings
        
    Returns:
        LLMProvider instance (OpenAI or Ollama)
        
    Raises:
        LLMError: If provider cannot be created or is unavailable
    """
    if config.provider == ProviderType.CLOUD:
        logger.info(f"☁️ Creating Cloud provider (OpenAI {config.model})")
        provider = OpenAIProvider(config)
    elif config.provider == ProviderType.LOCAL:
        logger.info(f"🏠 Creating Local provider (Ollama {config.ollama_model})")
        provider = OllamaProvider(config)
    else:
        raise LLMError(
            error_type=LLMErrorType.INVALID_REQUEST,
            message=f"Unknown provider type: {config.provider}",
            user_message="Invalid LLM provider configuration.",
            recoverable=False,
        )
    
    # Validate provider availability
    available, message = provider.is_available()
    if not available:
        raise LLMError(
            error_type=LLMErrorType.PROVIDER_UNAVAILABLE,
            message=message,
            user_message=message,
            recoverable=False,
            provider=provider.name,
        )
    
    logger.info(f"✅ {message}")
    return provider


def get_provider(config: LLMConfig | None = None) -> LLMProvider:
    """
    Get or create the global LLM provider instance.
    
    Args:
        config: Optional config to create new provider.
                If None and no provider exists, creates default cloud provider.
        
    Returns:
        LLMProvider instance
    """
    global _provider, _current_config
    
    # If config provided and different from current, create new provider
    if config is not None:
        if _current_config is None or config.provider != _current_config.provider:
            _provider = create_provider(config)
            _current_config = config
    
    # If no provider exists, create default
    if _provider is None:
        default_config = config or LLMConfig.for_cloud()
        _provider = create_provider(default_config)
        _current_config = default_config
    
    return _provider


def reset_provider() -> None:
    """Reset the global provider instance."""
    global _provider, _current_config
    _provider = None
    _current_config = None


def get_provider_from_env() -> LLMConfig | None:
    """
    Get LLM provider configuration from environment variable.
    
    Returns:
        LLMConfig if LLM_PROVIDER env var is set, None otherwise.
        
    Raises:
        LLMError: If LLM_PROVIDER has invalid value.
    """
    provider_env = os.environ.get(ENV_LLM_PROVIDER, "").lower().strip()
    
    if not provider_env:
        return None
    
    if provider_env == "cloud":
        logger.info(f"📝 LLM_PROVIDER={provider_env} - Using Cloud (OpenAI GPT-5)")
        model = os.environ.get("OPENAI_MODEL", "gpt-5")
        return LLMConfig.for_cloud(model=model)
    
    elif provider_env == "local":
        logger.info(f"📝 LLM_PROVIDER={provider_env} - Using Local (Ollama)")
        model = os.environ.get(ENV_LOCAL_MODEL, "llama3.2")
        base_url = os.environ.get(ENV_OLLAMA_HOST, "http://localhost:11434")
        return LLMConfig.for_local(model=model, base_url=base_url)
    
    else:
        raise LLMError(
            error_type=LLMErrorType.INVALID_REQUEST,
            message=f"Invalid LLM_PROVIDER value: '{provider_env}'",
            user_message=f"LLM_PROVIDER must be 'cloud' or 'local', got '{provider_env}'",
            recoverable=False,
        )


def auto_select_provider() -> LLMConfig:
    """
    Automatically select LLM provider based on environment or prompt user.
    
    Priority:
    1. LLM_PROVIDER environment variable (if set)
    2. Interactive prompt (if terminal is available)
    3. Default to cloud mode
    
    Returns:
        LLMConfig for the selected provider
    """
    # Check environment variable first
    env_config = get_provider_from_env()
    if env_config is not None:
        return env_config
    
    # Check if we're in an interactive terminal
    import sys
    if sys.stdin.isatty():
        return select_provider_interactive()
    
    # Default to cloud mode
    logger.info("📝 No LLM_PROVIDER set, defaulting to Cloud (OpenAI)")
    return LLMConfig.for_cloud()


def select_provider_interactive() -> LLMConfig:
    """
    Interactive provider selection for CLI startup.
    
    Returns:
        LLMConfig based on user selection
    """
    print("\n" + "=" * 50)
    print("🤖 SpectraVoice – LLM Provider Selection")
    print("=" * 50)
    print("\nChoose your LLM provider:")
    print("  [1] ☁️  Cloud (OpenAI GPT) - Requires internet & API key")
    print("  [2] 🏠 Local (Ollama) - Runs on your machine, no API key needed")
    print()
    
    while True:
        choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
        
        if choice == "" or choice == "1":
            return _configure_cloud_provider()
        elif choice == "2":
            return _configure_local_provider()
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def _configure_cloud_provider() -> LLMConfig:
    """Configure cloud (OpenAI) provider interactively."""
    print("\n☁️ Cloud Provider (OpenAI) selected")
    
    # Check for API key
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment.")
        print("   Set it with: export OPENAI_API_KEY=your-key")
    
    # Model selection (GPT‑5 family only)
    models = ["gpt-5", "gpt-5.1", "gpt-5.2"]
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        default = " (default)" if model == "gpt-5" else ""
        print(f"  [{i}] {model}{default}")
    
    while True:
        model_choice = input(f"Select model (1-{len(models)}) [default: 1]: ").strip()
        if model_choice == "":
            model = "gpt-5"
            break
        try:
            idx = int(model_choice) - 1
            if 0 <= idx < len(models):
                model = models[idx]
                break
        except ValueError:
            pass
        print(f"❌ Invalid choice. Enter 1-{len(models)}.")
    
    print(f"✅ Using Cloud provider with {model}")
    return LLMConfig.for_cloud(model=model)


def _configure_local_provider() -> LLMConfig:
    """Configure local (Ollama) provider interactively."""
    print("\n🏠 Local Provider (Ollama) selected")
    
    # Check if Ollama is running
    ollama_url = "http://localhost:11434"
    
    try:
        import httpx
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                
                if not models:
                    print("⚠️ No models found in Ollama.")
                    print("   Install a model with: ollama pull deepseek-r1:7b")
                    models = ["deepseek-r1:7b"]  # Default suggestion
                else:
                    print(f"\n✅ Ollama is running with {len(models)} model(s)")
                    print("\nAvailable models:")
                    for i, model in enumerate(models[:10], 1):  # Show first 10
                        print(f"  [{i}] {model}")
                    if len(models) > 10:
                        print(f"  ... and {len(models) - 10} more")
                
                # Model selection
                while True:
                    model_input = input(f"\nSelect model (1-{min(len(models), 10)}) or type name [default: 1]: ").strip()
                    if model_input == "":
                        model = models[0] if models else "deepseek-r1:7b"
                        break
                    try:
                        idx = int(model_input) - 1
                        if 0 <= idx < len(models):
                            model = models[idx]
                            break
                    except ValueError:
                        # User typed a model name
                        model = model_input
                        break
                    print(f"❌ Invalid choice.")
                
                print(f"✅ Using Local provider with {model}")
                return LLMConfig.for_local(model=model, base_url=ollama_url)
                
    except Exception as e:
        print(f"\n❌ Cannot connect to Ollama at {ollama_url}")
        print(f"   Error: {e}")
        print("\n📋 To start Ollama:")
        print("   1. Install: brew install ollama (macOS)")
        print("   2. Start: ollama serve")
        print("   3. Pull a model: ollama pull deepseek-r1:7b")
        print()
        
        # Allow user to proceed anyway
        proceed = input("Continue with default settings? (y/n) [default: n]: ").strip().lower()
        if proceed == "y":
            return LLMConfig.for_local(model="deepseek-r1:7b", base_url=ollama_url)
        
        # Fall back to cloud
        print("\n⬅️ Falling back to Cloud provider...")
        return _configure_cloud_provider()


def get_provider_status() -> dict[str, Any]:
    """Get the current provider's status."""
    if _provider is None:
        return {
            "initialized": False,
            "message": "No provider initialized",
        }
    
    status = _provider.get_status()
    status["initialized"] = True
    return status


def select_provider_gui() -> LLMConfig:
    """
    Show GUI selector for LLM provider selection.
    
    This function displays a small tkinter window with two options:
    - Cloud (OpenAI GPT)
    - Local (Ollama)
    
    If the GUI fails or is unavailable, falls back to terminal selection.
    
    Returns:
        LLMConfig based on user selection
    """
    from .gui_selector import (
        select_provider_gui as _gui_select,
        is_gui_available,
    )
    
    # Try GUI selection
    if is_gui_available():
        logger.debug("🖥️ GUI available, showing provider selector")
        config = _gui_select()
        if config is not None:
            provider_name = "Cloud (OpenAI)" if config.provider == ProviderType.CLOUD else "Local (Ollama)"
            model = config.model if config.provider == ProviderType.CLOUD else config.ollama_model
            logger.info(f"✅ GUI selection: {provider_name} with {model}")
            return config
        else:
            logger.debug("❌ GUI selection cancelled or failed")
    else:
        logger.debug("❌ GUI not available on this system")
    
    # Fallback to terminal selection
    logger.info("📝 Falling back to terminal selection")
    return select_provider_interactive()


def select_provider_with_fallback() -> LLMConfig:
    """
    Smart provider selection with GUI preference and multiple fallbacks.
    
    Priority:
    1. Environment variable (LLM_PROVIDER) - for non-interactive/CI environments
    2. GUI selector (if available and interactive)
    3. Terminal interactive selection (if stdin is a TTY)
    4. Default to cloud
    
    Returns:
        LLMConfig for the selected provider
    """
    # Check environment variable first (highest priority for non-interactive)
    env_config = get_provider_from_env()
    if env_config is not None:
        return env_config
    
    # Check if we're in an interactive environment
    import sys
    if not sys.stdin.isatty():
        # Non-interactive environment without LLM_PROVIDER set
        logger.info("📝 Non-interactive environment, defaulting to Cloud (OpenAI)")
        return LLMConfig.for_cloud()
    
    # Interactive environment - try GUI first, then terminal
    return select_provider_gui()
