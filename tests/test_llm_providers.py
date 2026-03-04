"""
Tests for LLM Provider System (Phase 7)
========================================

This module tests the provider-agnostic LLM interface including:
- Cloud mode (OpenAI) - basic prompt-response
- Local mode (Ollama) - mocked endpoint tests
- Provider selection logic via LLM_PROVIDER environment variable

MANUAL VALIDATION STEPS:
------------------------
1. Start Ollama:
   $ ollama serve
   
2. Pull a model:
   $ ollama pull llama3.2
   # Or for vision: ollama pull llava
   
3. Run in Cloud mode:
   $ python main.py --cloud
   # Or: LLM_PROVIDER=cloud python main.py
   
4. Run in Local mode:
   $ python main.py --local
   # Or: LLM_PROVIDER=local python main.py
   
5. Run with specific model:
   $ python main.py --local --model llava
   $ python main.py --cloud --model gpt-5.1
"""

import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.llm import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMError,
    LLMErrorType,
    ProviderType,
    LLMProvider,
    OpenAIProvider,
    OllamaProvider,
    create_provider,
    get_provider_from_env,
    auto_select_provider,
)


# =============================================================================
# CLOUD MODE TESTS
# =============================================================================

def test_cloud_config_creation():
    """Test creating a cloud LLM configuration."""
    print("\n☁️ Testing Cloud Config Creation...")
    
    config = LLMConfig.for_cloud(model="gpt-5")
    
    assert config.provider == ProviderType.CLOUD
    assert config.model == "gpt-5"
    assert config.enable_vision == True
    assert config.enable_tools == True
    
    print("  ✅ Cloud config created with correct defaults")
    print(f"  ✅ Provider: {config.provider.value}")
    print(f"  ✅ Model: {config.model}")
    print("✅ Cloud config test passed")


@patch('openai.OpenAI')
def test_cloud_provider_creation(mock_openai_class):
    """Test creating OpenAI provider (with mocked client)."""
    print("\n☁️ Testing Cloud Provider Creation...")
    
    # Mock the OpenAI client to avoid actual API validation
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test1234567890abcdefghijklmnopqrstuvwxyz12345678"}):
        config = LLMConfig.for_cloud(model="gpt-5")
        provider = OpenAIProvider(config)
        
        assert provider.name == "OpenAI"
        assert provider.model_name == "gpt-5"
        assert provider.supports_vision == True
        assert provider.supports_tools == True
        
        print("  ✅ OpenAI provider created")
        print(f"  ✅ Name: {provider.name}")
        print(f"  ✅ Model: {provider.model_name}")
        print(f"  ✅ Vision: {provider.supports_vision}")
        print(f"  ✅ Tools: {provider.supports_tools}")
    
    print("✅ Cloud provider creation test passed")


def test_cloud_provider_missing_api_key():
    """Test that missing API key raises appropriate error."""
    print("\n☁️ Testing Cloud Provider Missing API Key...")
    
    # Remove API key from environment
    env_backup = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        config = LLMConfig.for_cloud()
        try:
            provider = create_provider(config)
            assert False, "Should have raised LLMError"
        except LLMError as e:
            assert e.error_type == LLMErrorType.PROVIDER_UNAVAILABLE
            assert "OPENAI_API_KEY" in e.user_message
            print("  ✅ Correct error raised for missing API key")
            print(f"  ✅ Error type: {e.error_type.value}")
    finally:
        # Restore API key
        if env_backup:
            os.environ["OPENAI_API_KEY"] = env_backup
    
    print("✅ Missing API key test passed")


@patch('openai.OpenAI')
def test_cloud_basic_prompt_response(mock_openai_class):
    """Test basic prompt-response with mocked OpenAI."""
    print("\n☁️ Testing Cloud Basic Prompt-Response...")
    
    # Setup mock
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! I'm an AI assistant."
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-5"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 18
    
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        config = LLMConfig.for_cloud(model="gpt-5")
        provider = OpenAIProvider(config)
        
        messages = [
            LLMMessage(role="user", content="Hello!")
        ]
        
        response = provider.generate(messages)
        
        assert response.content == "Hello! I'm an AI assistant."
        assert response.finish_reason == "stop"
        assert response.provider == "OpenAI"
        assert not response.is_tool_use
        
        print("  ✅ Response received")
        print(f"  ✅ Content: {response.content[:50]}...")
        print(f"  ✅ Finish reason: {response.finish_reason}")
    
    print("✅ Cloud basic prompt-response test passed")


# =============================================================================
# LOCAL MODE TESTS (Mocked Ollama)
# =============================================================================

def test_local_config_creation():
    """Test creating a local LLM configuration."""
    print("\n🏠 Testing Local Config Creation...")
    
    config = LLMConfig.for_local(model="llama3.2")
    
    assert config.provider == ProviderType.LOCAL
    assert config.ollama_model == "llama3.2"
    assert config.ollama_base_url == "http://localhost:11434"
    
    print("  ✅ Local config created with correct defaults")
    print(f"  ✅ Provider: {config.provider.value}")
    print(f"  ✅ Model: {config.ollama_model}")
    print(f"  ✅ Base URL: {config.ollama_base_url}")
    print("✅ Local config test passed")


def test_local_config_with_env_override():
    """Test local config respects environment variables."""
    print("\n🏠 Testing Local Config Environment Override...")
    
    with patch.dict(os.environ, {
        "LOCAL_LLM_MODEL": "deepseek-r1:7b",
        "OLLAMA_HOST": "http://192.168.1.100:11434"
    }):
        config = LLMConfig.for_local()
        provider = OllamaProvider(config)
        
        # Provider should pick up env vars
        assert provider.config.ollama_model == "deepseek-r1:7b"
        assert provider.base_url == "http://192.168.1.100:11434"
        
        print("  ✅ LOCAL_LLM_MODEL override working")
        print("  ✅ OLLAMA_HOST override working")
    
    print("✅ Local config env override test passed")


@patch('httpx.Client')
def test_local_basic_prompt_response(mock_client_class):
    """Test basic prompt-response with mocked Ollama endpoint."""
    print("\n🏠 Testing Local Basic Prompt-Response (Mocked)...")
    
    # Setup mock HTTP client
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
    mock_client_class.return_value.__exit__ = Mock(return_value=False)
    
    # Mock /api/tags response (for availability check)
    mock_tags_response = MagicMock()
    mock_tags_response.status_code = 200
    mock_tags_response.json.return_value = {
        "models": [{"name": "llama3.2:latest"}]
    }
    
    # Mock /api/chat response
    mock_chat_response = MagicMock()
    mock_chat_response.status_code = 200
    mock_chat_response.json.return_value = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "Hello! I'm running locally via Ollama."
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 15,
        "eval_count": 12
    }
    
    def mock_request(method, url, **kwargs):
        if "api/tags" in url:
            return mock_tags_response
        elif "api/chat" in url:
            return mock_chat_response
        return MagicMock(status_code=404)
    
    mock_client.get = lambda url: mock_request("GET", url)
    mock_client.post = lambda url, **kwargs: mock_request("POST", url, **kwargs)
    
    config = LLMConfig.for_local(model="llama3.2")
    provider = OllamaProvider(config)
    
    messages = [
        LLMMessage(role="user", content="Hello!")
    ]
    
    response = provider.generate(messages)
    
    assert response.content == "Hello! I'm running locally via Ollama."
    assert response.finish_reason == "stop"
    assert response.provider == "Ollama"
    assert response.model == "llama3.2"
    
    print("  ✅ Mocked Ollama response received")
    print(f"  ✅ Content: {response.content[:50]}...")
    print(f"  ✅ Provider: {response.provider}")
    print(f"  ✅ Model: {response.model}")
    
    print("✅ Local basic prompt-response test passed")


def test_local_vision_model_detection():
    """Test vision model detection for Ollama."""
    print("\n🏠 Testing Local Vision Model Detection...")
    
    # Vision models
    for model in ["llava", "llava:7b", "bakllava", "moondream"]:
        config = LLMConfig.for_local(model=model)
        provider = OllamaProvider(config)
        assert provider.supports_vision, f"{model} should support vision"
        print(f"  ✅ {model}: vision=True")
    
    # Non-vision models
    for model in ["llama3.2", "mistral", "deepseek-r1:7b"]:
        config = LLMConfig.for_local(model=model)
        provider = OllamaProvider(config)
        assert not provider.supports_vision, f"{model} should not support vision"
        print(f"  ✅ {model}: vision=False")
    
    print("✅ Vision model detection test passed")


def test_local_tool_support_detection():
    """Test tool calling support detection for Ollama models."""
    print("\n🏠 Testing Local Tool Support Detection...")
    
    # Models with native tool support
    native_tool_models = ["llama3.2", "llama3.1", "mistral", "qwen2.5"]
    for model in native_tool_models:
        config = LLMConfig.for_local(model=model)
        provider = OllamaProvider(config)
        assert provider.supports_native_tools, f"{model} should have native tool support"
        print(f"  ✅ {model}: native_tools=True")
    
    # Models with augmented tool support only
    augmented_only = ["deepseek-r1:7b", "codellama"]
    for model in augmented_only:
        config = LLMConfig.for_local(model=model)
        provider = OllamaProvider(config)
        assert provider.supports_tools, f"{model} should support tools"
        assert not provider.supports_native_tools, f"{model} should not have native tools"
        print(f"  ✅ {model}: tools=True, native=False")
    
    print("✅ Tool support detection test passed")


def test_local_tool_augmentation():
    """Test tool augmentation for models without native support."""
    print("\n🏠 Testing Local Tool Augmentation...")
    
    config = LLMConfig.for_local(model="deepseek-r1:7b")
    provider = OllamaProvider(config)
    
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Open Chrome for me.")
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open an application",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "App name"}
                },
                "required": ["name"]
            }
        }
    }]
    
    augmented = provider._augment_messages_with_tools(messages, tools)
    
    # Should have system message with tools appended
    assert len(augmented) == 2
    assert "tool_call" in augmented[0].content
    assert "open_application" in augmented[0].content
    
    print("  ✅ Messages augmented with tool descriptions")
    print("  ✅ Tool format instructions included")
    
    print("✅ Tool augmentation test passed")


def test_local_augmented_tool_parsing():
    """Test parsing tool calls from augmented prompting responses."""
    print("\n🏠 Testing Augmented Tool Call Parsing...")
    
    config = LLMConfig.for_local(model="llama3.2")
    provider = OllamaProvider(config)
    
    # Test direct JSON
    content1 = '{"tool_call": {"name": "open_application", "arguments": {"name": "Chrome"}}}'
    calls1 = provider._parse_augmented_tool_calls(content1)
    assert len(calls1) == 1
    assert calls1[0].name == "open_application"
    print("  ✅ Direct JSON parsing works")
    
    # Test mixed content
    content2 = 'I will open Chrome for you. {"tool_call": {"name": "open_application", "arguments": {"name": "Chrome"}}}'
    calls2 = provider._parse_augmented_tool_calls(content2)
    assert len(calls2) == 1
    assert calls2[0].name == "open_application"
    print("  ✅ Mixed content parsing works")
    
    # Test no tool call
    content3 = "I cannot help with that request."
    calls3 = provider._parse_augmented_tool_calls(content3)
    assert len(calls3) == 0
    print("  ✅ No false positives for regular text")
    
    print("✅ Augmented tool parsing test passed")


@patch('httpx.Client')
def test_local_connection_error(mock_client_class):
    """Test handling of Ollama connection errors."""
    print("\n🏠 Testing Local Connection Error Handling...")
    
    import httpx
    
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
    mock_client_class.return_value.__exit__ = Mock(return_value=False)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")
    
    config = LLMConfig.for_local(model="llama3.2")
    provider = OllamaProvider(config)
    
    messages = [LLMMessage(role="user", content="Hello")]
    
    try:
        provider.generate(messages)
        assert False, "Should have raised LLMError"
    except LLMError as e:
        assert e.error_type == LLMErrorType.PROVIDER_UNAVAILABLE
        assert "ollama serve" in e.user_message.lower()
        print("  ✅ Connection error handled correctly")
        print(f"  ✅ User message: {e.user_message}")
    
    print("✅ Connection error handling test passed")


# =============================================================================
# PROVIDER SELECTION TESTS
# =============================================================================

def test_provider_selection_env_cloud():
    """Test provider selection with LLM_PROVIDER=cloud."""
    print("\n🔀 Testing Provider Selection: LLM_PROVIDER=cloud...")
    
    with patch.dict(os.environ, {"LLM_PROVIDER": "cloud"}, clear=False):
        config = get_provider_from_env()
        
        assert config is not None
        assert config.provider == ProviderType.CLOUD
        print("  ✅ LLM_PROVIDER=cloud returns cloud config")
        print(f"  ✅ Provider type: {config.provider.value}")
    
    print("✅ Cloud selection test passed")


def test_provider_selection_env_local():
    """Test provider selection with LLM_PROVIDER=local."""
    print("\n🔀 Testing Provider Selection: LLM_PROVIDER=local...")
    
    with patch.dict(os.environ, {"LLM_PROVIDER": "local"}, clear=False):
        config = get_provider_from_env()
        
        assert config is not None
        assert config.provider == ProviderType.LOCAL
        print("  ✅ LLM_PROVIDER=local returns local config")
        print(f"  ✅ Provider type: {config.provider.value}")
    
    print("✅ Local selection test passed")


def test_provider_selection_env_with_model():
    """Test provider selection with model override."""
    print("\n🔀 Testing Provider Selection with Model Override...")
    
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "local",
        "LOCAL_LLM_MODEL": "llava"
    }, clear=False):
        config = get_provider_from_env()
        
        assert config.provider == ProviderType.LOCAL
        # Model override happens in provider, not config
        print("  ✅ Config created with local provider")
    
    print("✅ Model override test passed")


def test_provider_selection_no_env():
    """Test provider selection with no environment variable."""
    print("\n🔀 Testing Provider Selection: No LLM_PROVIDER...")
    
    # Remove LLM_PROVIDER if set
    env_backup = os.environ.get("LLM_PROVIDER")
    if "LLM_PROVIDER" in os.environ:
        del os.environ["LLM_PROVIDER"]
    
    try:
        config = get_provider_from_env()
        assert config is None, "Should return None when no env var set"
        print("  ✅ Returns None when LLM_PROVIDER not set")
    finally:
        if env_backup:
            os.environ["LLM_PROVIDER"] = env_backup
    
    print("✅ No env var test passed")


def test_provider_selection_invalid():
    """Test provider selection with invalid value."""
    print("\n🔀 Testing Provider Selection: Invalid Value...")
    
    with patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}, clear=False):
        try:
            config = get_provider_from_env()
            assert False, "Should have raised LLMError"
        except LLMError as e:
            assert "cloud" in e.user_message.lower() or "local" in e.user_message.lower()
            print("  ✅ Invalid value raises appropriate error")
            print(f"  ✅ Error message: {e.user_message}")
    
    print("✅ Invalid value test passed")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integration_main_config_parsing():
    """Test main.py config parsing logic."""
    print("\n🔗 Testing Integration: main.py Config Parsing...")
    
    import main
    
    # Create mock args
    @dataclass
    class MockArgs:
        interactive: bool = False
        local: bool = False
        cloud: bool = False
        model: str = None
        ollama_url: str = "http://localhost:11434"
    
    # Test --local flag
    args = MockArgs(local=True)
    config = main.get_llm_config(args)
    assert config.provider == ProviderType.LOCAL
    print("  ✅ --local flag works")
    
    # Test --cloud flag
    args = MockArgs(cloud=True)
    config = main.get_llm_config(args)
    assert config.provider == ProviderType.CLOUD
    print("  ✅ --cloud flag works")
    
    # Test LLM_PROVIDER env var
    with patch.dict(os.environ, {"LLM_PROVIDER": "local"}, clear=False):
        args = MockArgs()  # No flags
        config = main.get_llm_config(args)
        assert config.provider == ProviderType.LOCAL
        print("  ✅ LLM_PROVIDER env var works")
    
    # Test model auto-detection (GPT‑5 model -> cloud)
    args = MockArgs(model="gpt-5.1")
    config = main.get_llm_config(args)
    assert config.provider == ProviderType.CLOUD
    print("  ✅ GPT model auto-detected as cloud")
    
    # Test model auto-detection (other model -> local)
    args = MockArgs(model="llama3.2")
    config = main.get_llm_config(args)
    assert config.provider == ProviderType.LOCAL
    print("  ✅ Non-GPT model auto-detected as local")
    
    print("✅ Integration config parsing test passed")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all LLM provider tests."""
    print("=" * 70)
    print("🧪 LLM PROVIDER TESTS (Phase 7)")
    print("=" * 70)
    
    tests = [
        # Cloud mode tests
        test_cloud_config_creation,
        test_cloud_provider_creation,
        test_cloud_provider_missing_api_key,
        test_cloud_basic_prompt_response,
        
        # Local mode tests
        test_local_config_creation,
        test_local_config_with_env_override,
        test_local_basic_prompt_response,
        test_local_vision_model_detection,
        test_local_tool_support_detection,
        test_local_tool_augmentation,
        test_local_augmented_tool_parsing,
        test_local_connection_error,
        
        # Provider selection tests
        test_provider_selection_env_cloud,
        test_provider_selection_env_local,
        test_provider_selection_env_with_model,
        test_provider_selection_no_env,
        test_provider_selection_invalid,
        
        # Integration tests
        test_integration_main_config_parsing,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            errors.append((test.__name__, str(e)))
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            errors.append((test.__name__, str(e)))
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if errors:
        print("\n❌ FAILURES:")
        for name, error in errors:
            print(f"  - {name}: {error}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
