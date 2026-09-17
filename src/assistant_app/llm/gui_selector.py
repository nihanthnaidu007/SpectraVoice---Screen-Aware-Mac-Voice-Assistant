"""
GUI-Based LLM Provider Selector
===============================
A small tkinter GUI pop-up for selecting between Cloud (OpenAI) and Local (Ollama) LLM providers.

Features:
- Clean, modern GUI with clear provider options
- Model selection dropdown for each provider
- Graceful fallback to terminal prompt if GUI fails
- Environment variable override for non-interactive runs

Usage:
    from assistant_app.llm.gui_selector import select_provider_gui
    
    config = select_provider_gui()  # Shows GUI, returns LLMConfig
"""

import os
from collections.abc import Callable

from .types import LLMConfig, ProviderType

# Default models for each provider (Cloud must be GPT‑5 or above)
CLOUD_MODELS = ["gpt-5", "gpt-5.1", "gpt-5.2"]
LOCAL_MODELS = ["llama3.2", "deepseek-r1:1.5b", "llava", "mistral", "qwen2.5"]


class LLMSelectorGUI:
    """
    A small GUI window for selecting LLM provider.
    
    Creates a centered pop-up with two clear options:
    - Cloud (GPT / OpenAI)
    - Local (Ollama)
    """
    
    def __init__(self):
        self.result: LLMConfig | None = None
        self.root = None
        self._available_local_models: list[str] = []
    
    def _fetch_ollama_models(self) -> list[str]:
        """Fetch available models from Ollama if running."""
        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                response = client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return models if models else LOCAL_MODELS
        except Exception:
            pass
        return LOCAL_MODELS
    
    def show(self) -> LLMConfig | None:
        """
        Display the GUI selector and return the selected configuration.
        
        Returns:
            LLMConfig if user made a selection, None if cancelled or error.
        """
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:
            return None
        
        # Fetch Ollama models in background
        self._available_local_models = self._fetch_ollama_models()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("SpectraVoice – LLM Provider")
        self.root.resizable(False, False)
        
        # Window sizing and centering
        window_width = 420
        window_height = 380
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Modern styling
        self.root.configure(bg="#1a1a2e")
        
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure styles
        style.configure("Title.TLabel", 
                       background="#1a1a2e", 
                       foreground="#ffffff",
                       font=("Helvetica", 16, "bold"))
        style.configure("Subtitle.TLabel",
                       background="#1a1a2e",
                       foreground="#888888",
                       font=("Helvetica", 10))
        style.configure("Option.TLabel",
                       background="#1a1a2e",
                       foreground="#ffffff",
                       font=("Helvetica", 11))
        style.configure("Cloud.TButton",
                       font=("Helvetica", 12, "bold"),
                       padding=(20, 12))
        style.configure("Local.TButton",
                       font=("Helvetica", 12, "bold"),
                       padding=(20, 12))
        style.configure("TCombobox",
                       font=("Helvetica", 10))
        
        # Main frame with padding
        main_frame = tk.Frame(self.root, bg="#1a1a2e", padx=30, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="🤖 SpectraVoice",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Select your LLM Provider",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # === Cloud Section ===
        cloud_frame = tk.Frame(main_frame, bg="#16213e", padx=15, pady=15)
        cloud_frame.pack(fill="x", pady=(0, 10))
        
        cloud_header = tk.Frame(cloud_frame, bg="#16213e")
        cloud_header.pack(fill="x")
        
        cloud_icon_label = tk.Label(
            cloud_header,
            text="☁️",
            font=("Helvetica", 20),
            bg="#16213e"
        )
        cloud_icon_label.pack(side="left")
        
        cloud_text_frame = tk.Frame(cloud_header, bg="#16213e")
        cloud_text_frame.pack(side="left", padx=(10, 0))
        
        cloud_title = tk.Label(
            cloud_text_frame,
            text="Cloud (OpenAI GPT)",
            font=("Helvetica", 12, "bold"),
            fg="#ffffff",
            bg="#16213e"
        )
        cloud_title.pack(anchor="w")
        
        cloud_desc = tk.Label(
            cloud_text_frame,
            text="Requires internet & API key",
            font=("Helvetica", 9),
            fg="#888888",
            bg="#16213e"
        )
        cloud_desc.pack(anchor="w")
        
        # Cloud model selector
        cloud_model_frame = tk.Frame(cloud_frame, bg="#16213e")
        cloud_model_frame.pack(fill="x", pady=(10, 10))
        
        cloud_model_label = tk.Label(
            cloud_model_frame,
            text="Model:",
            font=("Helvetica", 10),
            fg="#cccccc",
            bg="#16213e"
        )
        cloud_model_label.pack(side="left")
        
        self.cloud_model_var = tk.StringVar(value=CLOUD_MODELS[0])
        cloud_model_combo = ttk.Combobox(
            cloud_model_frame,
            textvariable=self.cloud_model_var,
            values=CLOUD_MODELS,
            state="readonly",
            width=18
        )
        cloud_model_combo.pack(side="left", padx=(10, 0))
        
        cloud_btn = tk.Button(
            cloud_frame,
            text="Use Cloud",
            font=("Helvetica", 11, "bold"),
            fg="#ffffff",
            bg="#4a90d9",
            activebackground="#3a7bc8",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            command=self._select_cloud
        )
        cloud_btn.pack(fill="x", pady=(5, 0), ipady=8)
        
        # === Local Section ===
        local_frame = tk.Frame(main_frame, bg="#16213e", padx=15, pady=15)
        local_frame.pack(fill="x", pady=(0, 10))
        
        local_header = tk.Frame(local_frame, bg="#16213e")
        local_header.pack(fill="x")
        
        local_icon_label = tk.Label(
            local_header,
            text="🏠",
            font=("Helvetica", 20),
            bg="#16213e"
        )
        local_icon_label.pack(side="left")
        
        local_text_frame = tk.Frame(local_header, bg="#16213e")
        local_text_frame.pack(side="left", padx=(10, 0))
        
        local_title = tk.Label(
            local_text_frame,
            text="Local (Ollama)",
            font=("Helvetica", 12, "bold"),
            fg="#ffffff",
            bg="#16213e"
        )
        local_title.pack(anchor="w")
        
        local_desc = tk.Label(
            local_text_frame,
            text="Runs locally, no API key needed",
            font=("Helvetica", 9),
            fg="#888888",
            bg="#16213e"
        )
        local_desc.pack(anchor="w")
        
        # Local model selector
        local_model_frame = tk.Frame(local_frame, bg="#16213e")
        local_model_frame.pack(fill="x", pady=(10, 10))
        
        local_model_label = tk.Label(
            local_model_frame,
            text="Model:",
            font=("Helvetica", 10),
            fg="#cccccc",
            bg="#16213e"
        )
        local_model_label.pack(side="left")
        
        # Use fetched models or defaults
        local_models = self._available_local_models or LOCAL_MODELS
        self.local_model_var = tk.StringVar(value=local_models[0])
        local_model_combo = ttk.Combobox(
            local_model_frame,
            textvariable=self.local_model_var,
            values=local_models,
            state="readonly",
            width=18
        )
        local_model_combo.pack(side="left", padx=(10, 0))
        
        local_btn = tk.Button(
            local_frame,
            text="Use Local",
            font=("Helvetica", 11, "bold"),
            fg="#ffffff",
            bg="#2ecc71",
            activebackground="#27ae60",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            command=self._select_local
        )
        local_btn.pack(fill="x", pady=(5, 0), ipady=8)
        
        # Footer hint
        hint_label = tk.Label(
            main_frame,
            text="💡 Tip: Set LLM_PROVIDER=cloud or local to skip this dialog",
            font=("Helvetica", 9),
            fg="#666666",
            bg="#1a1a2e"
        )
        hint_label.pack(pady=(10, 0))
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Bring to front
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        
        # Run event loop
        self.root.mainloop()
        
        return self.result
    
    def _select_cloud(self):
        """Handle Cloud selection."""
        model = self.cloud_model_var.get()
        self.result = LLMConfig.for_cloud(model=model)
        self.root.destroy()
    
    def _select_local(self):
        """Handle Local selection."""
        model = self.local_model_var.get()
        self.result = LLMConfig.for_local(model=model)
        self.root.destroy()
    
    def _on_close(self):
        """Handle window close (X button) - default to cloud."""
        self.result = None
        self.root.destroy()


def select_provider_gui() -> LLMConfig | None:
    """
    Show GUI selector and return the selected LLM configuration.
    
    Returns:
        LLMConfig if user made a selection, None if cancelled or GUI unavailable.
    """
    try:
        gui = LLMSelectorGUI()
        return gui.show()
    except Exception:
        # GUI failed, return None to trigger fallback
        return None


def select_provider_with_gui_fallback(
    terminal_fallback: Callable[[], LLMConfig] | None = None
) -> LLMConfig:
    """
    Attempt GUI selection with fallback to terminal prompt.
    
    Priority:
    1. Try GUI selector
    2. If GUI fails or is cancelled, use terminal fallback
    3. If no fallback provided, default to cloud
    
    Args:
        terminal_fallback: Optional function to call if GUI fails
        
    Returns:
        LLMConfig for the selected provider
    """
    # Try GUI first
    config = select_provider_gui()
    
    if config is not None:
        return config
    
    # GUI failed or was cancelled - try terminal fallback
    if terminal_fallback is not None:
        print("\n💡 GUI unavailable, using terminal selection...")
        return terminal_fallback()
    
    # No fallback - default to cloud (GPT‑5)
    print("\n💡 Using default Cloud provider (OpenAI GPT-5)")
    return LLMConfig.for_cloud(model="gpt-5")


def is_gui_available() -> bool:
    """
    Check if GUI (tkinter) is available on this system.
    
    Returns:
        True if tkinter can be imported and display is available.
    """
    try:
        import tkinter as tk
        # Try to create a hidden root to test display
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


def select_llm_provider() -> str:
    """
    Select LLM provider with environment variable override, GUI, or terminal fallback.
    
    This is the main entry point for provider selection as specified in the requirements.
    
    Priority:
    1. LLM_PROVIDER environment variable ("cloud" or "local")
    2. GUI pop-up selector (if available)
    3. Terminal prompt fallback
    
    Returns:
        "cloud" or "local"
    
    Example:
        mode = select_llm_provider()
        if mode == "cloud":
            # Use OpenAI
        else:
            # Use Ollama
    """
    # Step 1: Check environment variable first
    env_provider = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if env_provider in ("cloud", "local"):
        return env_provider
    
    # Step 2: Try GUI selector
    if is_gui_available():
        config = select_provider_gui()
        if config is not None:
            return "cloud" if config.provider == ProviderType.CLOUD else "local"
    
    # Step 3: Fallback to terminal prompt
    return _terminal_select_provider()


def _terminal_select_provider() -> str:
    """
    Terminal-based provider selection fallback.
    
    Returns:
        "cloud" or "local"
    """
    print("\n" + "=" * 50)
    print("🤖 SpectraVoice – LLM Provider Selection")
    print("=" * 50)
    print("\nSelect your LLM provider:")
    print("  [1] ☁️  Cloud (OpenAI GPT) - Requires internet & API key")
    print("  [2] 🏠 Local (Ollama) - Runs on your machine")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
            
            if choice == "" or choice == "1":
                print("✅ Selected: Cloud (OpenAI)")
                return "cloud"
            elif choice == "2":
                print("✅ Selected: Local (Ollama)")
                return "local"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except (EOFError, KeyboardInterrupt):
            # Non-interactive environment or user cancelled
            print("\n💡 Using default: Cloud (OpenAI)")
            return "cloud"
