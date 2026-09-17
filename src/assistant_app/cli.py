"""
SpectraVoice CLI — argument parsing and startup flow.
=====================================================

`main.py` at the repo root is a thin shim for this module, and the installed
`spectravoice` console script (see pyproject.toml) calls `cli.main()` directly.

Special modes handled here before the assistant starts:
- `--doctor`        run headless diagnostics (permissions, keys, config) and exit
- `--no-supervisor` run once, without the crash-restart supervisor
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

# Available voices and models
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Default LLM models (Cloud must be GPT‑5 or above)
CLOUD_MODELS = ["gpt-5", "gpt-5.1", "gpt-5.2"]
LOCAL_MODELS = ["llama3.2", "deepseek-r1:1.5b", "llava", "mistral", "qwen2.5"]


def parse_args(argv: list[str] | None = None):
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
  python main.py --debug                  Enable debug logging
  python main.py --doctor                 Check permissions, keys, and config
  python main.py --voice nova             Use Nova voice for TTS
  python main.py --whisper-model small    Use small Whisper model

LLM Provider Options:
  --cloud                   Use OpenAI GPT (requires OPENAI_API_KEY)
  --local                   Use Ollama local LLM (requires Ollama running)
  --interactive             Choose provider interactively at startup
  --model MODEL             Specify LLM model (e.g., gpt-5, llava, deepseek-r1:7b)
  --ollama-url URL          Ollama API URL (default: http://localhost:11434)
        """,
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
        help="LLM model name (e.g., gpt-5, llava, deepseek-r1:7b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml in the repo root)",
    )

    # Mode arguments
    parser.add_argument("--gui", action="store_true", help="Show GUI status window")
    parser.add_argument("--minimal", action="store_true", help="Run in minimal mode (fastest)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")

    # Doctor and supervision
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run headless diagnostics (mic/screen/accessibility permissions, API keys, "
        "config health) with fix-it links, then exit",
    )
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Run without the crash-restart supervisor (one run, even if supervisor.enabled "
        "is set in config.yaml)",
    )

    # Dictation (W1)
    parser.add_argument(
        "--dictation",
        action="store_true",
        help="Enable dictation mode: speech is cleaned (fillers removed, snippets "
        "expanded) and typed at the cursor instead of being sent to the assistant",
    )
    parser.add_argument(
        "--dictation-mode",
        type=str,
        default=None,
        choices=["push_to_talk", "vad"],
        help="Dictation activation: push_to_talk (hold the hotkey) or vad "
        "(continuous, utterance ends after a silence gap). Default: config.yaml "
        "dictation.activation",
    )

    # Voice and model arguments
    parser.add_argument(
        "--voice",
        type=str,
        default=None,  # None means use config.yaml voice.tts_voice
        choices=TTS_VOICES,
        help=f"TTS voice selection (default: config.yaml). Options: {', '.join(TTS_VOICES)}",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,  # None means use mode default
        choices=WHISPER_MODELS,
        help=f"Whisper model size (default: based on mode). Options: {', '.join(WHISPER_MODELS)}",
    )

    return parser.parse_args(argv)


def get_llm_config(args, cfg):
    """
    Build LLM configuration from command-line arguments, environment, or config.

    Priority:
    1. Explicit CLI flags (--cloud, --local, --interactive, --gui-select)
    2. LLM_PROVIDER environment variable
    3. --model override (cloud vs local inferred from model name)
    4. config.yaml `llm.provider` (cloud or local; `auto` falls through)
    5. GUI selector (default for interactive sessions)
    """
    from assistant_app.llm import LLMConfig
    from assistant_app.llm.factory import (
        get_provider_from_env,
        select_provider_gui,
        select_provider_interactive,
    )

    # Explicit CLI flags take priority
    if args.interactive:
        return select_provider_interactive()

    if getattr(args, "gui_select", False):
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

    # No CLI/env choice — config.yaml `llm.provider` decides.
    # "auto" keeps the legacy behavior: GUI selector with terminal fallback.
    if cfg.llm.provider == "local":
        return LLMConfig.for_local(model=cfg.llm.ollama_model, base_url=cfg.llm.ollama_url)
    if cfg.llm.provider == "cloud":
        return LLMConfig.for_cloud(model=cfg.llm.cloud_model)

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
            print("3. Pull the model:")
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


def _run_doctor(args) -> int:
    """Headless diagnostics: probe permissions and environment, print, exit."""
    from assistant_app.doctor import doctor_exit_code, run_doctor
    from assistant_app.utils.config import init_config

    config_manager = init_config(args.config)
    results = run_doctor(config_manager)
    return doctor_exit_code(results)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # --doctor is fully headless: config + report + exit, no assistant startup.
    if args.doctor:
        return _run_doctor(args)

    # Load config (config.yaml + VA_* env overrides) before logging setup so the
    # `logging:` section configures the rotating file handler.
    from assistant_app.utils.config import init_config

    config_manager = init_config(args.config)
    cfg = config_manager.config

    # Setup logging first (before any imports that might log), then install
    # crash capture so uncaught exceptions (including in background threads)
    # leave a traceback in the log file.
    from assistant_app.utils.logging_config import get_logger, install_crash_handlers, setup_logging

    setup_logging(
        debug=args.debug or cfg.debug,
        log_file=cfg.logging.file,
        max_bytes=cfg.logging.max_size_mb * 1024 * 1024,
        backup_count=cfg.logging.backup_count,
    )
    install_crash_handlers()
    logger = get_logger(__name__)

    # Surface config problems without blocking startup — defaults apply per key.
    for issue in config_manager.validate():
        logger.warning(f"⚠️ Config issue: {issue}")

    if args.debug or cfg.debug:
        logger.info("🐛 Debug mode enabled - verbose logging active")

    # Auto-restart: re-launch this entry point as a supervised child (the child
    # runs with --no-supervisor, so there is exactly one supervisor).
    if cfg.supervisor.enabled and not args.no_supervisor:
        from assistant_app.supervisor import main_entrypoint_command, supervise

        child_args = argv if argv is not None else sys.argv[1:]
        exit_code = supervise(main_entrypoint_command(child_args), config_manager)
        # Mirror shell convention for a child killed by a signal (128+N).
        return 128 + (-exit_code) if exit_code < 0 else exit_code

    # Get LLM configuration
    try:
        llm_config = get_llm_config(args, cfg)
    except Exception as e:
        logger.error(f"❌ Failed to configure LLM: {e}")
        print(f"\n❌ Configuration Error: {e}")
        print("\nUse --interactive for guided setup or check your settings.")
        return 1

    # Validate provider is available (fail-fast)
    provider, _provider_name, _model_name = validate_provider(llm_config, logger)

    # Determine mode: CLI flags win; config.yaml `mode:` is the default
    if args.minimal:
        mode = "minimal"
    elif args.gui:
        mode = "gui"
    else:
        mode = cfg.mode

    # Import and run (lazy import for faster startup)
    from assistant_app.services.spectravoice_assistant import SpectraVoiceAssistant

    assistant = SpectraVoiceAssistant(
        mode=mode,
        debug=args.debug or cfg.debug,
        voice=args.voice,
        whisper_model=args.whisper_model,
        llm_provider=provider,  # Pass validated provider directly
        config_manager=config_manager,
        dictation_enabled=args.dictation or cfg.dictation.enabled,
        dictation_activation=args.dictation_mode or cfg.dictation.activation,
    )
    assistant.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
