## SpectraVoice – Screen-Aware Mac Voice Assistant

SpectraVoice is a hands-free voice assistant for macOS that can see your screen, control your keyboard and mouse, search the web, and respond with fast, natural speech. It supports both OpenAI GPT‑5 (cloud) and Ollama (local) models, with barge‑in, short-term conversation memory, and a focus on honest, tool-driven actions.

---

## Features

- 🎤 **Voice Recognition**: Uses Whisper for speech-to-text
- 👁️ **Screen Analysis**: Can see and understand your desktop in real time
- 🧠 **Conversation Memory**: Remembers recent exchanges for context
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

Follow these steps in order. After this, you should be able to run SpectraVoice and talk to it.

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
| `--gui`           | Show GUI status window       | Off      |
| `--minimal`       | Minimal mode (fastest)       | Off      |
| `--debug`         | Enable debug logging         | Off      |
| `--voice`         | TTS voice selection          | shimmer  |
| `--whisper-model` | Whisper model size           | base     |

---

## License

MIT License @ 2026 Nihanth Naidu
