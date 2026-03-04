#!/bin/bash
# =============================================================================
# Ollama Setup Script for SpectraVoice
# =============================================================================
# This script helps set up Ollama for SpectraVoice's local LLM mode.
#
# Usage:
#   ./scripts/setup_ollama.sh              # Install default model (llama3.2)
#   ./scripts/setup_ollama.sh llava        # Install specific model
#   ./scripts/setup_ollama.sh --list       # List available models
#   ./scripts/setup_ollama.sh --status     # Check Ollama status
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model
DEFAULT_MODEL="llama3.2"

# Recommended models for the voice assistant
RECOMMENDED_MODELS=(
    "llama3.2"           # General purpose, good quality
    "llava"              # Vision support (can see screen)
    "deepseek-r1:1.5b"   # Fast, reasoning-focused
    "mistral"            # Good tool calling support
    "qwen2.5"            # Multilingual support
)

# Ollama API endpoint
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Ollama Setup for SpectraVoice${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

check_ollama_installed() {
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}❌ Ollama is not installed.${NC}"
        echo ""
        echo "To install Ollama:"
        echo -e "  ${YELLOW}macOS:${NC}   brew install ollama"
        echo -e "  ${YELLOW}Linux:${NC}   curl -fsSL https://ollama.ai/install.sh | sh"
        echo -e "  ${YELLOW}Windows:${NC} Download from https://ollama.ai/download"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}✅ Ollama is installed${NC}"
}

check_ollama_running() {
    if ! curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Ollama is not running.${NC}"
        echo ""
        echo "Starting Ollama..."
        ollama serve &
        sleep 3
        
        if ! curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
            echo -e "${RED}❌ Failed to start Ollama.${NC}"
            echo "Please start it manually with: ollama serve"
            exit 1
        fi
    fi
    echo -e "${GREEN}✅ Ollama is running at ${OLLAMA_HOST}${NC}"
}

list_installed_models() {
    echo -e "${BLUE}Installed models:${NC}"
    local models=$(curl -s "${OLLAMA_HOST}/api/tags" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    if not models:
        print('  (none)')
    else:
        for m in models:
            name = m.get('name', 'unknown')
            size = m.get('size', 0) / (1024**3)
            print(f'  - {name} ({size:.1f} GB)')
except:
    print('  (error reading models)')
" 2>/dev/null)
    echo "$models"
}

list_recommended() {
    echo ""
    echo -e "${BLUE}Recommended models for SpectraVoice:${NC}"
    echo ""
    echo "  Model              Description"
    echo "  ─────────────────  ────────────────────────────────────────"
    echo -e "  ${GREEN}llama3.2${NC}           General purpose, good balance of speed/quality"
    echo -e "  ${GREEN}llava${NC}              Vision support - can analyze your screen"
    echo -e "  ${GREEN}deepseek-r1:1.5b${NC}   Fast, reasoning-focused (1.5B params)"
    echo -e "  ${GREEN}mistral${NC}            Good tool/function calling support"
    echo -e "  ${GREEN}qwen2.5${NC}            Excellent multilingual support"
    echo ""
    echo "For vision/screen analysis, use: llava"
    echo "For fastest responses, use: deepseek-r1:1.5b"
    echo ""
}

pull_model() {
    local model="${1:-$DEFAULT_MODEL}"
    echo ""
    echo -e "${BLUE}Pulling model: ${model}${NC}"
    echo "This may take a few minutes depending on your internet speed..."
    echo ""
    
    if ollama pull "$model"; then
        echo ""
        echo -e "${GREEN}✅ Successfully pulled ${model}${NC}"
        echo ""
        echo "You can now run the assistant with:"
        echo -e "  ${YELLOW}python main.py --local --model ${model}${NC}"
    else
        echo ""
        echo -e "${RED}❌ Failed to pull ${model}${NC}"
        echo "Check if the model name is correct or try a different model."
        exit 1
    fi
}

show_status() {
    print_header
    check_ollama_installed
    check_ollama_running
    echo ""
    list_installed_models
    list_recommended
}

show_help() {
    echo "Usage: $0 [command] [model]"
    echo ""
    echo "Commands:"
    echo "  (none)      Pull the default model (llama3.2)"
    echo "  --list      List installed and recommended models"
    echo "  --status    Check Ollama installation status"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Pull default model (llama3.2)"
    echo "  $0 llava              # Pull llava (vision model)"
    echo "  $0 deepseek-r1:1.5b   # Pull DeepSeek 1.5B"
    echo "  $0 --list             # Show available models"
    echo ""
}

# Main
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    --list|-l)
        print_header
        check_ollama_running
        list_installed_models
        list_recommended
        ;;
    --status|-s)
        show_status
        ;;
    "")
        print_header
        check_ollama_installed
        check_ollama_running
        echo ""
        list_installed_models
        pull_model "$DEFAULT_MODEL"
        ;;
    *)
        print_header
        check_ollama_installed
        check_ollama_running
        pull_model "$1"
        ;;
esac
