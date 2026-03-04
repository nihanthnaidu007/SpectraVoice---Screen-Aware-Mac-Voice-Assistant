"""Tests for Function Calling / Tool Use (Task 14)."""

import os
import sys

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.tools.tool_executor import (
    ToolExecutor, 
    ToolResult, 
    SafetyConfig,
    TOOL_SCHEMAS,
    APP_NAME_MAPPINGS,
    BLOCKED_COMMANDS,
    SAFE_COMMAND_PREFIXES
)
from assistant_app.tools.safety import (
    SafetyValidator,
    RiskLevel,
    HIGH_RISK_PATTERNS,
    BLOCKED_OPERATIONS
)


def test_tool_schemas():
    """Test that tool schemas are properly formatted for OpenAI."""
    print("\n📋 Testing Tool Schemas...")
    
    schemas = ToolExecutor.get_tool_schemas()
    assert len(schemas) == 13, f"Expected 13 tool schemas, got {len(schemas)}"
    
    required_tools = [
        "open_application",
        "type_text", 
        "click_element",
        "click_at_cursor",
        "move_mouse",
        "drag_mouse",
        "get_mouse_position",
        "search_files",
        "run_command",
        "keyboard_shortcut",
        "scroll",
        "take_screenshot",
        "open_url"
    ]
    
    for tool in required_tools:
        found = any(s["function"]["name"] == tool for s in schemas)
        assert found, f"Missing tool schema: {tool}"
        print(f"  ✅ {tool} schema valid")
    
    # Verify schema structure
    for schema in schemas:
        assert "type" in schema, "Schema missing 'type'"
        assert schema["type"] == "function", "Schema type must be 'function'"
        assert "function" in schema, "Schema missing 'function'"
        func = schema["function"]
        assert "name" in func, "Function missing 'name'"
        assert "description" in func, "Function missing 'description'"
        assert "parameters" in func, "Function missing 'parameters'"
    
    print("  ✅ All schemas have correct structure")
    print(f"✅ Tool schemas test passed ({len(schemas)} tools)")


def test_app_name_mappings():
    """Test application name mappings."""
    print("\n📱 Testing App Name Mappings...")
    
    assert "chrome" in APP_NAME_MAPPINGS
    assert APP_NAME_MAPPINGS["chrome"] == "Google Chrome"
    assert "vscode" in APP_NAME_MAPPINGS
    assert APP_NAME_MAPPINGS["vscode"] == "Visual Studio Code"
    
    print(f"  ✅ {len(APP_NAME_MAPPINGS)} app mappings configured")
    print("✅ App name mappings test passed")


def test_safety_config():
    """Test safety configuration."""
    print("\n🔒 Testing Safety Configuration...")
    
    config = SafetyConfig()
    
    # Check defaults
    assert config.require_confirmation == False
    assert len(config.blocked_commands) > 0
    assert len(config.safe_command_prefixes) > 0
    assert config.max_command_timeout == 60
    assert config.dry_run == False
    
    print(f"  ✅ {len(config.blocked_commands)} blocked commands")
    print(f"  ✅ {len(config.safe_command_prefixes)} safe command prefixes")
    print("✅ Safety config test passed")


def test_blocked_commands():
    """Test that dangerous commands are blocked."""
    print("\n🚫 Testing Blocked Commands...")
    
    executor = ToolExecutor()
    
    dangerous_commands = [
        "rm -rf /",
        "sudo rm something",
        ":(){ :|:& };:",  # Fork bomb
        "shutdown now",
    ]
    
    for cmd in dangerous_commands:
        result = executor.execute("run_command", {"command": cmd})
        assert not result.success, f"Dangerous command should be blocked: {cmd}"
        print(f"  ✅ Blocked: {cmd[:30]}...")
    
    print("✅ Blocked commands test passed")


def test_safe_commands():
    """Test that safe commands are allowed."""
    print("\n✅ Testing Safe Commands...")
    
    executor = ToolExecutor()
    
    safe_commands = [
        ("ls -la", True),
        ("pwd", True),
        ("date", True),
        ("whoami", True),
        ("echo hello", True),
    ]
    
    for cmd, should_succeed in safe_commands:
        result = executor.execute("run_command", {"command": cmd})
        if should_succeed:
            assert result.success, f"Safe command should succeed: {cmd}"
            print(f"  ✅ Allowed: {cmd}")
        else:
            assert not result.success, f"Command should fail: {cmd}"
    
    print("✅ Safe commands test passed")


def test_tool_executor_dry_run():
    """Test dry run mode."""
    print("\n🔒 Testing Dry Run Mode...")
    
    config = SafetyConfig(dry_run=True)
    executor = ToolExecutor(config)
    
    result = executor.execute("open_application", {"name": "Chrome"})
    assert result.success
    assert "[DRY RUN]" in result.message
    print("  ✅ Dry run mode working")
    
    print("✅ Dry run test passed")


def test_safety_validator():
    """Test the safety validator."""
    print("\n🛡️ Testing Safety Validator...")
    
    validator = SafetyValidator(enable_confirmation=False)
    
    # Test low risk operation
    allowed, reason, risk = validator.validate("open_application", {"name": "Safari"})
    assert allowed
    assert risk == RiskLevel.LOW
    print(f"  ✅ Low risk: open Safari")
    
    # Test blocked operation
    allowed, reason, risk = validator.validate("run_command", {"command": "sudo rm -rf /"})
    assert not allowed
    assert risk == RiskLevel.CRITICAL
    print(f"  ✅ Blocked: sudo rm -rf /")
    
    print("✅ Safety validator test passed")


def test_tool_result():
    """Test ToolResult dataclass."""
    print("\n📦 Testing ToolResult...")
    
    # Success result
    result = ToolResult(success=True, message="Done", data={"key": "value"})
    assert result.success
    assert result.message == "Done"
    assert result.data == {"key": "value"}
    assert result.error is None
    
    # Failure result
    result = ToolResult(success=False, message="Failed", error="Something went wrong")
    assert not result.success
    assert result.error == "Something went wrong"
    
    print("  ✅ ToolResult creation working")
    print("✅ ToolResult test passed")


def test_execution_log():
    """Test execution logging."""
    print("\n📝 Testing Execution Log...")
    
    executor = ToolExecutor()
    executor.clear_execution_log()
    
    # Execute some commands
    executor.execute("run_command", {"command": "echo test"})
    executor.execute("run_command", {"command": "pwd"})
    
    log = executor.get_execution_log()
    assert len(log) == 2
    assert log[0]["tool"] == "run_command"
    
    print(f"  ✅ {len(log)} operations logged")
    print("✅ Execution log test passed")


def run_all_tests():
    """Run all function calling tests."""
    print("=" * 60)
    print("🧪 FUNCTION CALLING TESTS (Task 14)")
    print("=" * 60)
    
    tests = [
        test_tool_schemas,
        test_app_name_mappings,
        test_safety_config,
        test_blocked_commands,
        test_safe_commands,
        test_tool_executor_dry_run,
        test_safety_validator,
        test_tool_result,
        test_execution_log,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
