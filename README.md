## SpectraVoice – Screen-Aware Mac Voice Assistant

SpectraVoice is a hands-free voice assistant for macOS that can see your screen, control your keyboard and mouse, search the web, and respond with fast, natural speech. It supports both OpenAI GPT‑5 (cloud) and Ollama (local) models, with barge‑in, short-term conversation memory, and a focus on honest, tool-driven actions.

---

## Features

- 🎤 **Voice Recognition**: Uses Whisper for speech-to-text
- 👁️ **Screen Analysis**: Can see and understand your desktop in real time
- 🧠 **Conversation Memory**: Remembers recent exchanges for context
- 🗂️ **History Q&A**: Ask questions across your stored meeting transcripts — local-first (your configured Ollama model), cloud only behind an explicit consent flag, answers rendered in the history window (never spoken)
- 🗣️ **Natural Speech**: High-quality text-to-speech with multiple voices
- 🔧 **Function Calling**: Executes real actions on your computer via tools
- 🌐 **Web Search**: Real-time web search for current information
- ☁️ **Cloud LLM**: OpenAI GPT‑5 family for maximum quality and vision support
- 🏠 **Local LLM**: Ollama for privacy and offline use (DeepSeek, LLaMA, etc.)
- ⚡ **Multiple Modes**: Terminal, GUI, and minimal modes
- 🐛 **Debug Mode**: Performance metrics and verbose logging
- 🎙️ **Customizable**: Choose TTS voice and Whisper model size
- 📍 **Status Indicator**: Small on-screen indicator showing assistant state

---

## Voice Commands (Function Calling)

Control your computer with voice:

| Command              | Example                                      |
|----------------------|----------------------------------------------|
| **Open Apps**        | “Open Chrome”, “Open Notes”, “Open Spotify” |
| **Browse Web**       | “Go to YouTube”, “Open github.com”          |
| **Type Text**        | “Type hello world”                          |
| **Keyboard Shortcuts** | “Copy this”, “Paste”, “Save”              |
| **File Search**      | “Search for my resume”, “Find presentation files” |
| **Screenshots**      | “Take a screenshot”                         |
| **Scroll**           | “Scroll down”, “Scroll up”                  |
| **Terminal**         | “Run ls command”, “Show current directory”  |

---

## Installation & Setup (step by step)

### Quick start — 3 steps

```bash
# 1. Get the code
git clone https://github.com/nihanthnaidu007/SpectraVoice---Screen-Aware-Mac-Voice-Assistant.git
cd SpectraVoice---Screen-Aware-Mac-Voice-Assistant

# 2. One-command bootstrap (venv, system deps, pip install, .env, doctor check)
./scripts/bootstrap.sh

# 3. Run it
python main.py
```

`python main.py --doctor` re-checks everything any time: microphone, screen
recording, and accessibility permissions (with System Settings fix-it links),
API keys, Ollama reachability, and config health. Full details:
[docs/SETUP.md](docs/SETUP.md).

The manual steps below describe what the bootstrap automates.

### 1. Get the code

Using git (recommended):

```bash
git clone https://github.com/<your-username>/spectravoice.git
cd spectravoice
```

Or download the ZIP from GitHub and unzip it, then open a terminal and `cd` into that folder.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of your terminal prompt.

### 3. Install system dependencies

These are needed for audio and media handling.

**macOS (Apple Silicon/Intel):**

```bash
brew install portaudio ffmpeg
```

**Ubuntu/Debian (if you ever run it there):**

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg
```

(Windows is not the main target, but you would install FFmpeg and PortAudio separately.)

### 4. Install Python dependencies

With the virtual environment active and inside the project folder:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Create `.env` and add API keys

In the project root, create a `.env` file:

```bash
# If there is an example file
cp .env.example .env

# Or create manually
touch .env
```

Open `.env` and add:

```env
OPENAI_API_KEY="your_openai_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"  # Optional: for web search
```

- `OPENAI_API_KEY` is **required** for cloud mode (GPT‑5).
- `TAVILY_API_KEY` is optional but improves web search.
- Do **not** commit `.env` to GitHub.

### 6. Grant macOS permissions (very important)

On macOS, you must give your terminal/IDE permissions:

1. Open **System Settings → Privacy & Security**.
2. Under **Microphone**, enable access for:
   - Terminal / iTerm / your IDE.
3. Under **Screen Recording**, enable access for:
   - Terminal / your IDE.
4. Under **Accessibility**, enable access for:
   - Terminal / your IDE (needed for keyboard/mouse control).

You may need to quit and reopen the terminal or IDE after changing these.

---

## LLM Provider Modes: Cloud vs Local (Ollama)

SpectraVoice supports two LLM backends:

| Feature       | Cloud (OpenAI GPT‑5)            | Local (Ollama)                  |
|--------------|----------------------------------|---------------------------------|
| **Provider** | OpenAI GPT‑5 family             | Ollama (llama3.2, llava, etc.) |
| **Quality**  | ⭐⭐⭐⭐⭐ Best                      | ⭐⭐⭐ Good                       |
| **Vision**   | ✅ Full support                  | ⚠️ Requires llava model        |
| **Tool Calling** | ✅ Native                    | ⚠️ Model-dependent             |
| **Privacy**  | ❌ Data sent to OpenAI           | ✅ 100% local                  |
| **Offline**  | ❌ Requires internet             | ✅ Works offline               |
| **Cost**     | 💰 API usage fees                | 🆓 Free                        |

### Cloud Mode (Default) – OpenAI GPT‑5

Uses OpenAI’s GPT‑5 models for the highest quality and best tool + vision support.

```bash
# Default usage (GUI provider selection, GPT‑5 cloud by default)
python main.py

# Force cloud mode explicitly
python main.py --cloud

# Use a specific GPT‑5 family model
python main.py --cloud --model gpt-5.1
```

**Pros:**

- ✅ Best quality and accuracy  
- ✅ Full vision support (screen analysis)  
- ✅ Native function calling for all tools  
- ✅ Fastest responses  

**Cons:**

- ❌ Requires `OPENAI_API_KEY` in `.env`  
- ❌ Requires internet connection  
- ❌ API usage costs  

### Local Mode – Ollama

Runs models locally on your machine using Ollama.

```bash
# Basic local mode (uses llama3.2 by default)
python main.py --local

# With vision support (llava)
python main.py --local --model llava

# Fast reasoning model
python main.py --local --model deepseek-r1:1.5b
```

**Pros:**

- ✅ No OpenAI key required  
- ✅ Works completely offline  
- ✅ Data never leaves your machine  
- ✅ Free to use  

**Cons:**

- ⚠️ Vision support requires specific models (`llava`, etc.)  
- ⚠️ Tool calling behavior depends on the model  
- ⚠️ Often slower and less capable than GPT‑5  

### Automatic Provider Selection

You can let SpectraVoice choose based on environment variables:

```bash
# In your shell or .env file
export LLM_PROVIDER=cloud   # Always use Cloud (OpenAI)
export LLM_PROVIDER=local   # Always use Local (Ollama)

# Then:
python main.py
```

**Selection priority (from highest to lowest):**

1. CLI flags: `--cloud`, `--local`, `--interactive`
2. `LLM_PROVIDER` environment variable
3. GUI pop-up selector
4. Terminal prompt fallback
5. Default to cloud mode

---

## GUI Provider Selection (Startup Window)

When you run:

```bash
python main.py
```

with no flags and `LLM_PROVIDER` not set, SpectraVoice shows a small window:

### How It Looks

```text
┌─────────────────────────────────────┐
│  🤖 SpectraVoice                    │
│  Select your LLM Provider           │
│                                     │
│  ☁️ Cloud (OpenAI GPT‑5)            │
│  Requires internet & API key        │
│  Model: [gpt-5 ▾]                  │
│  [      Use Cloud       ]           │
│                                     │
│  🏠 Local (Ollama)                  │
│  Runs locally, no API key needed    │
│  Model: [llama3.2 ▾]                │
│  [      Use Local       ]           │
│                                     │
│  💡 Set LLM_PROVIDER to skip this   │
└─────────────────────────────────────┘
```

### Skip the GUI

To always use one provider:

```bash
# Always use Cloud (no GUI)
export LLM_PROVIDER=cloud
python main.py

# Always use Local (no GUI)
export LLM_PROVIDER=local
python main.py

# Or inline
LLM_PROVIDER=cloud python main.py
LLM_PROVIDER=local python main.py
```

### Headless / Non-Interactive Environments

If tkinter is not available (e.g. SSH, Docker), SpectraVoice falls back to a simple terminal prompt.

---

## Local LLM Setup (Ollama) – Detailed

### Step 1: Install Ollama

**macOS:**

```bash
brew install ollama
```

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**

Download from `https://ollama.ai/download`.

### Step 2: Start Ollama server

```bash
ollama serve
```

Keep this running while you use SpectraVoice.

### Step 3: Pull models

```bash
# Helper script (from project root)
./scripts/setup_ollama.sh

# Or pull manually:
ollama pull llama3.2
ollama pull llava
ollama pull deepseek-r1:1.5b
ollama pull mistral
```

### Step 4: Run SpectraVoice with local models

```bash
# Basic local
python main.py --local

# Vision (screen-aware) with llava
python main.py --local --model llava

# Faster model
python main.py --local --model deepseek-r1:1.5b
```

---

## Usage Modes

### Terminal Mode (recommended)

```bash
python main.py
```

- No extra window, uses your terminal
- Full screen analysis, minimal overhead

### GUI Mode

```bash
python main.py --gui
```

- Shows a status window
- Press `q` or `ESC` to quit the GUI

### Minimal Mode (fastest)

```bash
python main.py --minimal
```

- Shorter responses and lighter models
- Best when you care about speed

### Debug Mode

```bash
python main.py --debug
```

- Extra logs and timing
- Helpful when something is not working

### Custom voice and Whisper model

```bash
# Change TTS voice
python main.py --voice nova

# Change Whisper model
python main.py --whisper-model small
```

Available Whisper sizes: `tiny`, `base` (default), `small`, `medium`, `large`.

---

## On-Screen Status Indicator

SpectraVoice shows a small, simple indicator to show what it’s doing:

- **Listening** (green dot)
- **Thinking** (processing)
- **Speaking**

You can configure it:

```bash
# Disable the indicator entirely
export ASSISTANT_INDICATOR_ENABLED=false

# Change position (default: top-right)
export ASSISTANT_INDICATOR_POSITION=bottom-left
# Options: top-right, top-left, bottom-right, bottom-left
```

If tkinter is not installed, SpectraVoice can still run without the indicator.

---

## How to Use (simple flow)

1. Open a terminal, `cd` into the project, and activate `.venv`.
2. Run `python main.py` (or `python main.py --cloud` / `--local`).
3. Wait for it to say it’s **ready**.
4. Speak a command like:
   - “Open Chrome”
   - “Go to youtube.com”
   - “Search for AI videos on YouTube”
   - “Scroll down”
5. When you’re done, press **Ctrl + C** in the terminal (or `q` in GUI mode) to exit.

---

## Configuration (config.yaml)

All runtime settings live in `config.yaml` at the repo root (or pass `--config <path>`).
Every key has a built-in default, unknown keys are warned about and ignored, and any
`VA_<SECTION>_<KEY>` environment variable overrides the file (e.g. `VA_VOICE_WHISPER_MODEL=tiny`,
`VA_MICROPHONE_INPUT_DEVICE=2`). Values that are invalid per `validate()` log warnings at
startup and fall back to defaults — startup never crashes on a bad config file.

| Section | What it controls |
|---|---|
| `mode` | Default run mode when no `--gui` / `--minimal` flag is given (`terminal`, `gui`, `minimal`) |
| `modes.<name>` | Per-mode overrides: `screen_quality`, `screen_scale`, `whisper_model`, `max_tokens` |
| `voice` | TTS voice, Whisper model, language, `speech_rate` (0.25–4.0) |
| `screen` | Capture `quality` (1–100), `scale_factor` (0.1–1.0), `refresh_interval`, `cache_duration` |
| `barge_in` | Interruption behavior: `enabled`, `min_confidence` (0–1), `min_words`, `min_chars`, timeouts |
| `microphone` | `input_device` index (`null` = system default), `energy_threshold`, `pause_threshold`, ambient-noise seconds, `inline_transcription` |
| `tts` | `output_device` index (`null` = system default), `hd_quality` |
| `llm` | `provider`: `cloud`, `local`, or `auto` (interactive/GUI selection, the default) |
| `logging` | Log `level`, `file`, `max_size_mb`, `backup_count` |

**Audio devices:** find your device indices with `python scripts/list_audio_devices.py`, then set
`microphone.input_device` / `tts.output_device`. A configured device that disappears (e.g. an
unplugged headset) falls back to the system default with a log warning instead of crashing.

**Latency:** Whisper transcription runs on a background worker by default — the recognizer
loop hands utterances off immediately instead of blocking for the full transcription time.
Set `microphone.inline_transcription: true` to restore the pre-W1 inline behavior.
Measure the difference with `python scripts/benchmark_transcription_latency.py`.

---

## Troubleshooting (quick)

**Microphone not working**

- Check macOS microphone permission.
- Make sure the correct input device is selected in System Settings.

**No screen context / errors about screen recording**

- Ensure Screen Recording permission is turned on for your terminal or IDE.

**Mouse/keyboard not controlled**

- Ensure Accessibility permission is turned on for your terminal or IDE.

**OpenAI errors**

- Confirm `OPENAI_API_KEY` is set correctly in `.env`.
- Check that your key has access to GPT‑5.
- Ensure you have internet.

**High CPU usage**

- Try `--minimal` mode.
- Close other heavy applications.

---

## Project Structure

```text
SpectraVoice/
├── main.py                  # Entry point (run this)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (you create this)
├── README.md                # Documentation
├── scripts/
│   └── setup_ollama.sh      # Helper script for Ollama
├── tests/
│   ├── test_function_calling.py
│   └── test_llm_providers.py
└── src/
    └── assistant_app/
        ├── __init__.py              # Package exports
        ├── core/
        │   ├── assistant.py         # Core assistant logic
        │   └── conversation.py      # Conversation memory
        ├── llm/
        │   ├── __init__.py
        │   ├── base.py              # Provider base class
        │   ├── types.py             # LLMConfig, message/response types
        │   ├── factory.py           # Provider creation and selection
        │   ├── openai_provider.py   # Cloud (OpenAI GPT‑5) provider
        │   └── ollama_provider.py   # Local (Ollama) provider
        ├── io/
        │   ├── audio/
        │   │   ├── tts.py           # Text-to-speech
        │   │   └── voice_detector.py# Voice detection & barge‑in
        │   ├── vision/
        │   │   └── screen_capture.py# Screen capture service
        │   └── indicator.py         # On-screen status indicator
        ├── services/
        │   └── spectravoice_assistant.py  # Main orchestrator
        ├── tools/
        │   ├── web_search.py        # Web search integration
        │   ├── tool_executor.py     # Tool execution for actions
        │   └── safety.py            # Safety checks for tools
        └── utils/
            ├── logging_config.py    # Logging setup
            ├── config.py            # Config management
            └── error_handler.py     # Error handling
```

---

## Running Tests

```bash
# With Python directly
python tests/test_llm_providers.py
python tests/test_function_calling.py

# If you have pytest installed
pytest tests/ -v
```

---

## CLI Reference

### LLM Provider options

| Flag            | Description                          | Default  |
|----------------|--------------------------------------|----------|
| `--cloud`      | Use Cloud LLM (OpenAI GPT‑5)         | Off      |
| `--local`      | Use Local LLM (Ollama)               | Off      |
| `--interactive`| Terminal-based provider selection    | Off      |
| `--gui-select` | Force GUI provider selector          | On (default) |
| `--model`      | LLM model name                       | gpt‑5 / llama3.2 |
| `--ollama-url` | Ollama API URL                       | `http://localhost:11434` |

### General options

| Flag              | Description                  | Default  |
|-------------------|------------------------------|----------|
| `--config`        | Path to config file          | `config.yaml` in repo root |
| `--gui`           | Show GUI status window       | Off      |
| `--minimal`       | Minimal mode (fastest)       | Off      |
| `--debug`         | Enable debug logging         | Off      |
| `--doctor`        | Headless diagnostics: permission probes, API keys, config health; prints fix-it links and exits | Off |
| `--no-supervisor` | Run once without the crash-restart supervisor (even if `supervisor.enabled` in config.yaml) | Off |
| `--voice`         | TTS voice selection          | `voice.tts_voice` from config (shimmer) |
| `--whisper-model` | Whisper model size           | base     |

---

## License

MIT License @ 2026 Nihanth Naidu
