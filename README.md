# ## SpectraVoice – Screen-Aware Mac Voice Assistant

SpectraVoice is a hands-free voice assistant for macOS that can see your screen, control your keyboard and mouse, search the web, and respond with fast, natural speech. It supports both OpenAI (cloud) and Ollama (local) models, with barge‑in, conversation memory, and a focus on honest, tool-driven actions.

## Features

- 🎤 **Voice Recognition**: Powered by OpenAI Whisper (local processing)
- 👁️ **Screen Analysis**: Can see and understand your desktop in real-time
- 🧠 **Conversation Memory**: Remembers last 10 exchanges for context
- 🗣️ **Natural Speech**: High-quality text-to-speech with 6 voice options
- 🔧 **Function Calling**: Execute actions on your computer via voice commands
- 🌐 **Web Search**: Real-time web search for current information
- ☁️ **Cloud LLM**: OpenAI GPT-4o for maximum quality and vision support
- 🏠 **Local LLM**: Ollama for privacy and offline use (DeepSeek, LLaMA, etc.)
- ⚡ **Multiple Modes**: Terminal, GUI, and minimal modes for different needs
- 🐛 **Debug Mode**: Performance metrics and verbose logging
- 🎙️ **Customizable**: Choose TTS voice and Whisper model size
- 📍 **Status Indicator**: Small on-screen indicator showing assistant state

## Voice Commands (Function Calling)

Control your computer with voice:

| Command | Example |
|---------|---------|
| **Open Apps** | "Open Chrome", "Open Notes", "Open Spotify" |
| **Browse Web** | "Go to YouTube", "Open github.com" |
| **Type Text** | "Type hello world" |
| **Keyboard Shortcuts** | "Copy this", "Paste", "Save" |
| **File Search** | "Search for my resume", "Find presentation files" |
| **Screenshots** | "Take a screenshot" |
| **Scroll** | "Scroll down", "Scroll up" |
| **Terminal** | "Run ls command", "Show current directory" |

## Setup

### 1. API Key

Create a `.env` file in the project directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # Optional: for web search
```

### 2. System Dependencies

**macOS (Apple Silicon/Intel):**
```bash
brew install portaudio ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

**Windows:**
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH
- PyAudio should install automatically with pip

### 3. Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (all exact versions included)
pip install -U pip
pip install -r requirements.txt
```

### 4. macOS Permissions (Important!)

Grant these permissions in System Preferences → Privacy & Security:
- **Microphone**: Allow access for Terminal/IDE
- **Screen Recording**: Allow access for screen capture
- **Accessibility**: Allow access for keyboard/mouse automation (function calling)

## LLM Provider Modes: Cloud vs Local (Ollama)

SpectraVoice supports two LLM backends:

| Feature | Cloud (OpenAI) | Local (Ollama) |
|---------|----------------|----------------|
| **Provider** | OpenAI GPT-4o | Ollama (llama3.2, llava, etc.) |
| **Quality** | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐ Good |
| **Vision** | ✅ Full support | ⚠️ Requires llava model |
| **Tool Calling** | ✅ Native | ⚠️ Model-dependent |
| **Privacy** | ❌ Data sent to OpenAI | ✅ 100% local |
| **Offline** | ❌ Requires internet | ✅ Works offline |
| **Cost** | 💰 API usage fees | 🆓 Free |

### Cloud Mode (Default) – OpenAI GPT‑5

Uses OpenAI's GPT‑5 models for maximum quality and full feature support.

```bash
python main.py                    # Uses GPT-5 by default
python main.py --cloud            # Explicit cloud mode
python main.py --model gpt-5.1    # Use a different GPT-5 family model
```

**Pros:**
- ✅ Best quality and accuracy
- ✅ Full vision support (screen analysis)
- ✅ Native function calling for all tools
- ✅ Fastest response times

**Cons:**
- ❌ Requires `OPENAI_API_KEY` in `.env`
- ❌ Requires internet connection
- ❌ API usage costs

### Local Mode - Ollama

Uses Ollama to run LLMs locally on your machine for privacy and offline use.

```bash
python main.py --local            # Uses llama3.2 by default
python main.py --local --model llava  # Use vision-capable model
python main.py --local --model deepseek-r1:1.5b  # Fast model
```

**Pros:**
- ✅ No API key required
- ✅ Works completely offline
- ✅ Data never leaves your machine
- ✅ Free to use

**Cons:**
- ⚠️ Vision support requires specific models (llava, bakllava)
- ⚠️ Tool calling support varies by model
- ⚠️ Slower than cloud on most hardware

### Automatic Provider Selection

The assistant can automatically select the provider based on environment variables:

```bash
# Set in your shell or .env file
export LLM_PROVIDER=cloud   # Always use Cloud (OpenAI)
export LLM_PROVIDER=local   # Always use Local (Ollama)

# Then just run:
python main.py  # Will use the provider from LLM_PROVIDER
```

**Selection Priority:**
1. CLI flags (`--cloud`, `--local`, `--interactive`) - highest priority
2. `LLM_PROVIDER` environment variable
3. GUI pop-up selector (if available)
4. Terminal prompt fallback
5. Default to Cloud mode

---

## GUI Provider Selection (Startup Window)

When you run `python main.py` without any flags or environment variables, a **small GUI window** appears allowing you to choose between Cloud and Local LLM providers.

### How It Works

```
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

Set the `LLM_PROVIDER` environment variable to bypass the GUI:

```bash
# Always use Cloud (no GUI)
export LLM_PROVIDER=cloud
python main.py

# Always use Local (no GUI)
export LLM_PROVIDER=local
python main.py

# Or inline:
LLM_PROVIDER=cloud python main.py
LLM_PROVIDER=local python main.py
```

### Headless/Non-Interactive Environments

If tkinter is unavailable (e.g., SSH sessions, Docker containers), the selector automatically falls back to a terminal prompt:

```
==================================================
🤖 SpectraVoice – LLM Provider Selection
==================================================

Select your LLM provider:
  [1] ☁️  Cloud (OpenAI GPT) - Requires internet & API key
  [2] 🏠 Local (Ollama) - Runs on your machine

Enter choice (1 or 2) [default: 1]: 
```

### GUI Requirements

The GUI uses Python's built-in `tkinter`. If not available:

**macOS:**
```bash
brew install python-tk
```

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

---

## Local LLM Setup (Ollama)

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
Download from https://ollama.ai/download

### Step 2: Start Ollama Server

```bash
ollama serve
```

> **Note:** Keep this running in a separate terminal. The assistant will fail-fast with helpful instructions if Ollama is not running.

### Step 3: Pull a Model

```bash
# Use our helper script (recommended):
./scripts/setup_ollama.sh

# Or manually pull models:
ollama pull llama3.2        # General purpose (3.2B params)
ollama pull llava           # Vision support (recommended for screen analysis)
ollama pull deepseek-r1:1.5b  # Fast, reasoning-focused
ollama pull mistral         # Good balance of speed/quality
```

### Step 4: Run the Assistant

```bash
# Basic local mode
python main.py --local

# With vision support
python main.py --local --model llava

# With specific model
python main.py --local --model deepseek-r1:1.5b
```

### Environment Variables for Local LLM

```bash
# Override default local model
export LOCAL_LLM_MODEL="llama3.2"

# Override Ollama API URL (for remote Ollama servers)
export OLLAMA_HOST="http://localhost:11434"

# Auto-select local mode
export LLM_PROVIDER="local"
```

### Model Capabilities

| Model | Vision | Tool Calling | Speed | Quality |
|-------|--------|--------------|-------|---------|
| `llama3.2` | ❌ | ✅ Native | Fast | Good |
| `llama3.1` | ❌ | ✅ Native | Medium | Better |
| `llava` | ✅ | ⚠️ Limited | Medium | Good |
| `mistral` | ❌ | ✅ Native | Fast | Good |
| `deepseek-r1:1.5b` | ❌ | ⚠️ Augmented | Very Fast | Basic |
| `qwen2.5` | ❌ | ✅ Native | Medium | Good |

> **Tool Calling Note:** Models without native tool support use "augmented prompting" - the assistant describes available tools in the prompt and parses JSON responses. This works but is less reliable than native tool calling.

### Interactive Mode

```bash
python main.py --interactive
```

- Guides you through provider selection at startup
- Lists available Ollama models
- Helps configure settings
- Useful for first-time setup

---

## Usage

### Terminal Mode (Recommended)
```bash
python main.py
```
- No visual window - prevents mirror effect
- Full screen analysis capability
- Lowest resource usage

### GUI Mode
```bash
python main.py --gui
```
- Shows status window with listening indicator
- Press 'q' or ESC to quit

### Minimal Mode (Fastest)
```bash
python main.py --minimal
```
- Fastest startup and response
- Lower quality for speed
- Best for low-end devices

### Debug Mode
```bash
python main.py --debug
```
- Verbose logging with timestamps
- Performance metrics after each interaction
- Useful for troubleshooting

### Custom Voice
```bash
python main.py --voice nova
```
Available voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` (default)

### Custom Whisper Model
```bash
python main.py --whisper-model small
```
Available models: `tiny`, `base` (default), `small`, `medium`, `large`
- Larger models = better accuracy, slower processing
- Smaller models = faster, less accurate

### Combine Options
```bash
python main.py --debug --voice echo --whisper-model small
```

## On-Screen Status Indicator

The assistant displays a small, unobtrusive indicator at the screen edge showing its current state:

- **📍 Listening** (green dot) - Ready for voice input
- **📍 Thinking** (orange dot) - Processing your request
- **📍 Live** - Assistant is running

### Indicator Features
- Always-on-top positioning (visible over other apps)
- Draggable - click and drag to reposition
- Pulsing dot for visual feedback
- Automatic hide on shutdown

### Configuration
```bash
# Disable the indicator
export ASSISTANT_INDICATOR_ENABLED=false

# Change position (default: top-right)
export ASSISTANT_INDICATOR_POSITION=bottom-left  # top-right, top-left, bottom-right, bottom-left
```

### Requirements
The indicator uses `tkinter` which is typically bundled with Python. If not available:

**macOS:**
```bash
brew install python-tk
```

> Note: If tkinter is unavailable, the assistant works normally without the visual indicator.

## How to Use

1. Run the assistant with your preferred mode
2. Speak naturally - the assistant is always listening
3. Ask about anything on your screen or general questions
4. Press Ctrl+C (terminal) or 'q' (GUI) to quit

## Example Questions

- "What's on my screen?"
- "Help me with this application"
- "What does this error message mean?"
- "Can you read this text for me?"
- "What should I click next?"
- "What did I just ask you?" (memory test)
- "Tell me more about that" (follow-up)

## Performance Modes

| Mode | Startup | CPU | Memory | Best For |
|------|---------|-----|--------|----------|
| **Terminal** | 1.5s | Low | 80MB | Daily use |
| **GUI** | 2.0s | Medium | 100MB | Visual feedback |
| **Minimal** | 0.8s | Lowest | 50MB | Low-end devices |

## Troubleshooting

### Microphone Issues
- Ensure microphone permissions are enabled in System Preferences
- Check that your microphone is set as the default input device

### Audio Output Issues
- Check speaker/headphone connections
- Verify system audio output settings

### API Errors
- Ensure your OpenAI API key is valid
- Check that you have sufficient API credits
- Verify internet connection

### High CPU Usage
- Use `--minimal` mode for lower resource usage
- Close other resource-intensive applications

## Project Structure

```
SpectraVoice/
├── main.py                              # Entry point (run this)
├── requirements.txt                     # Python dependencies (exact versions)
├── .env                                 # API keys (create this)
├── README.md                            # Documentation
├── scripts/
│   └── setup_ollama.sh                 # Helper script for Ollama setup
├── tests/
│   ├── test_function_calling.py        # Function calling tests
│   └── test_llm_providers.py           # LLM provider tests (Phase 7)
└── src/
    └── assistant_app/
        ├── __init__.py                  # Package exports
        ├── core/
        │   ├── assistant.py             # Vision AI with Function Calling
        │   └── conversation.py          # Conversation memory
        ├── llm/                          # LLM Provider Module
        │   ├── __init__.py              # Module exports
        │   ├── base.py                  # Provider-agnostic interface
        │   ├── types.py                 # LLMConfig, LLMMessage, LLMResponse
        │   ├── factory.py               # Provider creation & selection
        │   ├── openai_provider.py       # Cloud (OpenAI GPT) implementation
        │   └── ollama_provider.py       # Local (Ollama) implementation
        ├── io/
        │   ├── audio/
        │   │   ├── tts.py               # Text-to-speech
        │   │   └── voice_detector.py    # Smart voice detection
        │   ├── vision/
        │   │   └── screen_capture.py    # Screen capture service
        │   └── indicator.py             # On-screen status indicator
        ├── services/
        │   └── spectravoice_assistant.py# Main SpectraVoice orchestration controller
        ├── tools/
        │   ├── web_search.py            # Tavily web search integration
        │   ├── tool_executor.py         # Function calling tool executor
        │   └── safety.py                # Safety validation for tools
        └── utils/
            ├── logging_config.py        # Structured logging setup
            ├── config.py                # Configuration management
            └── error_handler.py         # Error handling utilities
```

## Running Tests

```bash
# Run all LLM provider tests
python tests/test_llm_providers.py

# Run function calling tests
python tests/test_function_calling.py

# Run with pytest (if installed)
pytest tests/ -v
```

### Test Coverage

| Test Suite | Coverage |
|------------|----------|
| `test_llm_providers.py` | Cloud config, Local config, Provider selection, Tool augmentation |
| `test_function_calling.py` | Tool schemas, Safety validation, Command execution |

---

## CLI Reference

### LLM Provider Options
| Flag | Description | Default |
|------|-------------|---------|
| `--cloud` | Use Cloud LLM (OpenAI GPT) | Off |
| `--local` | Use Local LLM (Ollama) | Off |
| `--interactive` | Terminal-based provider selection | Off |
| `--gui-select` | Force GUI provider selector | ✓ (default) |
| `--model` | LLM model name | gpt-5 / llama3.2 |
| `--ollama-url` | Ollama API URL | http://localhost:11434 |

> **Note:** If no flag is provided and `LLM_PROVIDER` env is not set, the GUI selector appears by default.

### General Options
| Flag | Description | Default |
|------|-------------|---------|
| `--gui` | Show GUI status window | Off |
| `--minimal` | Minimal mode (fastest) | Off |
| `--debug` | Enable debug logging | Off |
| `--voice` | TTS voice selection | shimmer |
| `--whisper-model` | Whisper model size | base |

## License

MIT License