#!/usr/bin/env python3
"""
SpectraVoice – Screen-Aware Mac Voice Assistant
===============================================
A powerful screen-aware voice assistant for macOS that can see your screen,
control your computer, and help with tasks.
Supports both Cloud (OpenAI GPT-5 family) and Local (Ollama) LLM providers.

Usage:
    python main.py                          # Shows GUI pop-up for LLM selection
    python main.py --local                  # Use local Ollama LLM
    python main.py --local --model llava    # Use specific Ollama model
    python main.py --cloud                  # Explicitly use Cloud (OpenAI)
    python main.py --interactive            # Terminal-based provider selection
    python main.py --gui-select             # Force GUI provider selector
    python main.py --gui                    # GUI mode with status window
    python main.py --minimal                # Minimal mode (fastest)
    python main.py --debug                  # Enable debug logging
    python main.py --voice nova             # Use Nova voice for TTS
    python main.py --whisper-model small    # Use small Whisper model

LLM Provider Selection:
    By default, a GUI pop-up appears for selecting Cloud or Local LLM.
    Set LLM_PROVIDER=cloud or LLM_PROVIDER=local to skip the GUI.
"""

import sys
import pathlib

# Ensure src/ is importable when running `python main.py` from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import argparse
from dotenv import load_dotenv

load_dotenv()

# Available voices and models
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Default LLM models (Cloud must be GPT‑5 or above)
CLOUD_MODELS = ["gpt-5", "gpt-5.1", "gpt-5.2"]
LOCAL_MODELS = ["llama3.2", "deepseek-r1:1.5b", "llava", "mistral", "qwen2.5"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SpectraVoice – Screen-Aware Mac Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Run with Cloud LLM (default)
  python main.py --local                  Run with Local LLM (Ollama)
  python main.py --local --model llava    Run with specific Ollama model
  python main.py --interactive            Interactive provider selection
  python main.py --gui                    Run with status window
  python main.py --minimal                Run in minimal mode (fastest)
  python main.py --debug                  Enable verbose debug logging
  python main.py --voice nova             Use Nova voice for TTS
  python main.py --whisper-model small    Use small Whisper model for better accuracy

LLM Provider Options:
  --cloud                   Use OpenAI GPT (requires OPENAI_API_KEY)
  --local                   Use Ollama local LLM (requires Ollama running)
  --interactive             Choose provider interactively at startup
  --model MODEL             Specify LLM model (e.g., gpt-5, llava, deepseek-r1:7b)
  --ollama-url URL          Ollama API URL (default: http://localhost:11434)
        """
    )
    
    # LLM Provider arguments
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--cloud", action="store_true", help="Use Cloud LLM (OpenAI GPT)")
    llm_group.add_argument("--local", action="store_true", help="Use Local LLM (Ollama)")
    llm_group.add_argument("--interactive", action="store_true", help="Terminal-based provider selection")
    llm_group.add_argument("--gui-select", action="store_true", help="GUI-based provider selection (default behavior)")
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (e.g., gpt-5, llava, deepseek-r1:7b)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    # Mode arguments
    parser.add_argument("--gui", action="store_true", help="Show GUI status window")
    parser.add_argument("--minimal", action="store_true", help="Run in minimal mode (fastest)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Voice and model arguments
    parser.add_argument(
        "--voice", 
        type=str, 
        default="shimmer",
        choices=TTS_VOICES,
        help=f"TTS voice selection (default: shimmer). Options: {', '.join(TTS_VOICES)}"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,  # None means use mode default
        choices=WHISPER_MODELS,
        help=f"Whisper model size (default: based on mode). Options: {', '.join(WHISPER_MODELS)}"
    )
    
    return parser.parse_args()


def get_llm_config(args):
    """
    Build LLM configuration from command-line arguments or environment.
    
    Priority:
    1. Explicit CLI flags (--cloud, --local, --interactive, --gui-select)
    2. LLM_PROVIDER environment variable
    3. GUI selector (default for interactive sessions)
    4. Default to cloud mode (for non-interactive)
    """
    from assistant_app.llm import LLMConfig, ProviderType
    from assistant_app.llm.factory import (
        select_provider_interactive, 
        select_provider_gui,
        get_provider_from_env,
    )
    
    # Explicit CLI flags take priority
    if args.interactive:
        return select_provider_interactive()
    
    if getattr(args, 'gui_select', False):
        return select_provider_gui()
    
    if args.local:
        model = args.model or "llama3.2"
        return LLMConfig.for_local(
            model=model,
            base_url=args.ollama_url,
        )
    
    if args.cloud:
        model = args.model or "gpt-5"
        return LLMConfig.for_cloud(model=model)
    
    # No explicit flag - check environment variable
    import os
    env_provider = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if env_provider:
        return get_provider_from_env()
    
    # No env var either - check for model override
    if args.model:
        # User specified a model, try to detect which provider
        cloud_models = {"gpt-5", "gpt-5.1", "gpt-5.2"}
        if args.model in cloud_models or args.model.startswith("gpt-5"):
            return LLMConfig.for_cloud(model=args.model)
        else:
            # Assume it's a local model
            return LLMConfig.for_local(model=args.model, base_url=args.ollama_url)
    
    # Default behavior: Show GUI selector (with terminal fallback)
    # This is the new default when running `python main.py` without flags
    return select_provider_gui()


def validate_provider(llm_config, logger):
    """
    Validate the LLM provider is available and fail-fast if not.
    
    Returns:
        Tuple of (provider_name, model_name) for logging
        
    Raises:
        SystemExit: If provider is not available
    """
    from assistant_app.llm import LLMError
    from assistant_app.llm.factory import create_provider
    
    is_local = llm_config.provider.value == "local"
    provider_name = "Local (Ollama)" if is_local else "Cloud (OpenAI)"
    model_name = llm_config.ollama_model if is_local else llm_config.model
    
    # Validate provider is reachable
    try:
        provider = create_provider(llm_config)
        
        # Log success with appropriate message
        if is_local:
            logger.info(f"🏠 Using Local LLM (Ollama, model={model_name})")
        else:
            logger.info(f"☁️ Using Cloud LLM (GPT, model={model_name})")
        
        return provider, provider_name, model_name
        
    except LLMError as e:
        logger.error(f"❌ {e.user_message}")
        print(f"\n❌ LLM Provider Error: {e.user_message}")
        
        if is_local:
            print("\n" + "=" * 50)
            print("📋 Ollama Setup Instructions:")
            print("=" * 50)
            print("1. Install Ollama:")
            print("   brew install ollama  # macOS")
            print("   # Or visit: https://ollama.ai/download")
            print()
            print("2. Start Ollama server:")
            print("   ollama serve")
            print()
            print(f"3. Pull the model:")
            print(f"   ollama pull {model_name}")
            print()
            print("4. Run the assistant again:")
            print(f"   python main.py --local --model {model_name}")
            print("=" * 50)
        else:
            print("\n📋 Cloud Setup:")
            print("  - Ensure OPENAI_API_KEY is set in .env or environment")
            print("  - Check your internet connection")
            print("  - Verify API key is valid at platform.openai.com")
        
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging first (before any imports that might log)
    from assistant_app.utils.logging_config import setup_logging, get_logger
    setup_logging(debug=args.debug)
    logger = get_logger(__name__)
    
    if args.debug:
        logger.info("🐛 Debug mode enabled - verbose logging active")
    
    # Get LLM configuration
    try:
        llm_config = get_llm_config(args)
    except Exception as e:
        logger.error(f"❌ Failed to configure LLM: {e}")
        print(f"\n❌ Configuration Error: {e}")
        print("\nUse --interactive for guided setup or check your settings.")
        sys.exit(1)
    
    # Validate provider is available (fail-fast)
    provider, provider_name, model_name = validate_provider(llm_config, logger)
    
    # Determine mode
    if args.minimal:
        mode = "minimal"
    elif args.gui:
        mode = "gui"
    else:
        mode = "terminal"
    
    # Import and run (lazy import for faster startup)
    from assistant_app.services.spectravoice_assistant import SpectraVoiceAssistant
    
    assistant = SpectraVoiceAssistant(
        mode=mode, 
        debug=args.debug,
        voice=args.voice,
        whisper_model=args.whisper_model,
        llm_provider=provider,  # Pass validated provider directly
    )
    assistant.run()


if __name__ == "__main__":
    main()
