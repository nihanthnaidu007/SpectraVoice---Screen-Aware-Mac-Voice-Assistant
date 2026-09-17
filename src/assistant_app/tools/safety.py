"""Safety Validation System for Tool Execution."""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from assistant_app.utils.logging_config import get_logger


class RiskLevel(Enum):
    """Risk levels for tool operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyRule:
    """A safety rule for tool validation."""
    name: str
    description: str
    risk_level: RiskLevel
    check: Callable[[str, dict], bool]  # (tool_name, args) -> is_dangerous


# High-risk operations that require confirmation
HIGH_RISK_PATTERNS = {
    "run_command": [
        ("rm", RiskLevel.CRITICAL, "File deletion command"),
        ("mv", RiskLevel.HIGH, "File move command"),
        ("cp", RiskLevel.MEDIUM, "File copy command"),
        ("chmod", RiskLevel.HIGH, "Permission change command"),
        ("chown", RiskLevel.HIGH, "Ownership change command"),
        ("curl", RiskLevel.MEDIUM, "Network request"),
        ("wget", RiskLevel.MEDIUM, "Network download"),
        ("pip install", RiskLevel.MEDIUM, "Package installation"),
        ("brew install", RiskLevel.MEDIUM, "Package installation"),
        ("npm install", RiskLevel.MEDIUM, "Package installation"),
    ],
    "open_application": [
        ("Terminal", RiskLevel.LOW, "Opening terminal"),
        ("System Preferences", RiskLevel.LOW, "Opening system settings"),
    ],
    "type_text": [
        ("password", RiskLevel.HIGH, "Potential password entry"),
        ("secret", RiskLevel.HIGH, "Potential secret entry"),
        ("api_key", RiskLevel.HIGH, "Potential API key entry"),
        ("token", RiskLevel.HIGH, "Potential token entry"),
    ],
    "keyboard_shortcut": [
        (["command", "q"], RiskLevel.MEDIUM, "Quit application"),
        (["command", "w"], RiskLevel.LOW, "Close window"),
        (["command", "delete"], RiskLevel.HIGH, "Delete operation"),
    ],
}


# Blocked operations (never allowed)
BLOCKED_OPERATIONS = {
    "run_command": [
        "sudo",
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /*",
        "mkfs",
        "dd if=/dev/zero",
        "> /dev/sda",
        ":(){ :|:& };:",
        "fork bomb",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init 0",
        "init 6",
    ],
}


class SafetyValidator:
    """Validates tool operations for safety."""
    
    def __init__(
        self,
        enable_confirmation: bool = True,
        audit_log_path: str | None = None,
        rate_limit_per_minute: int = 30,
        confirmation_callback: Callable[[str, RiskLevel], bool] | None = None
    ):
        self.logger = get_logger(__name__)
        self.enable_confirmation = enable_confirmation
        self.audit_log_path = audit_log_path or os.path.expanduser(
            "~/.spectravoice/tool_audit.log"
        )
        self.rate_limit = rate_limit_per_minute
        self.confirmation_callback = confirmation_callback
        
        # Rate limiting state
        self._operation_times: list[float] = []
        
        # Ensure audit log directory exists
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
    
    def validate(self, tool_name: str, arguments: dict) -> tuple[bool, str, RiskLevel]:
        """
        Validate a tool operation.
        
        Returns:
            tuple: (is_allowed, reason, risk_level)
        """
        # Check rate limiting
        if not self._check_rate_limit():
            return False, "Rate limit exceeded. Please wait before executing more commands.", RiskLevel.HIGH
        
        # Check if operation is blocked
        is_blocked, reason = self._check_blocked(tool_name, arguments)
        if is_blocked:
            self._log_blocked_operation(tool_name, arguments, reason)
            return False, reason, RiskLevel.CRITICAL
        
        # Assess risk level
        risk_level, risk_reason = self._assess_risk(tool_name, arguments)
        
        # For high/critical risk, check if confirmation is needed
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and self.enable_confirmation:
            if self.confirmation_callback:
                confirmed = self.confirmation_callback(
                    f"⚠️ {risk_reason}\nTool: {tool_name}\nArgs: {arguments}",
                    risk_level
                )
                if not confirmed:
                    return False, "Operation cancelled by user.", risk_level
            else:
                self.logger.warning(f"⚠️ High-risk operation: {risk_reason}")
        
        # Log the operation
        self._log_operation(tool_name, arguments, risk_level, "allowed")
        
        return True, "Operation allowed", risk_level
    
    def _check_blocked(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """Check if the operation is in the blocked list."""
        if tool_name not in BLOCKED_OPERATIONS:
            return False, ""
        
        blocked_patterns = BLOCKED_OPERATIONS[tool_name]
        
        if tool_name == "run_command":
            command = arguments.get("command", "").lower()
            for pattern in blocked_patterns:
                if pattern.lower() in command:
                    return True, f"Blocked operation: {pattern}"
        
        return False, ""
    
    def _assess_risk(self, tool_name: str, arguments: dict) -> tuple[RiskLevel, str]:
        """Assess the risk level of an operation."""
        if tool_name not in HIGH_RISK_PATTERNS:
            return RiskLevel.LOW, "Standard operation"
        
        patterns = HIGH_RISK_PATTERNS[tool_name]
        
        if tool_name == "run_command":
            command = arguments.get("command", "").lower()
            for pattern, risk, description in patterns:
                if pattern.lower() in command:
                    return risk, description
        
        elif tool_name == "type_text":
            text = arguments.get("text", "").lower()
            for pattern, risk, description in patterns:
                if pattern.lower() in text:
                    return risk, description
        
        elif tool_name == "keyboard_shortcut":
            keys = [k.lower() for k in arguments.get("keys", [])]
            for pattern, risk, description in patterns:
                pattern_keys = [k.lower() for k in pattern]
                if pattern_keys == keys:
                    return risk, description
        
        elif tool_name == "open_application":
            app_name = arguments.get("name", "").lower()
            for pattern, risk, description in patterns:
                if pattern.lower() in app_name:
                    return risk, description
        
        return RiskLevel.LOW, "Standard operation"
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Remove old timestamps
        self._operation_times = [t for t in self._operation_times if t > minute_ago]
        
        # Check limit
        if len(self._operation_times) >= self.rate_limit:
            return False
        
        # Add current operation
        self._operation_times.append(current_time)
        return True
    
    def _log_operation(
        self, 
        tool_name: str, 
        arguments: dict, 
        risk_level: RiskLevel, 
        status: str
    ) -> None:
        """Log an operation to the audit file."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "arguments": arguments,
            "risk_level": risk_level.value,
            "status": status
        }
        
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write audit log: {e}")
    
    def _log_blocked_operation(
        self, 
        tool_name: str, 
        arguments: dict, 
        reason: str
    ) -> None:
        """Log a blocked operation."""
        self.logger.warning(f"🚫 BLOCKED: {tool_name} - {reason}")
        self._log_operation(tool_name, arguments, RiskLevel.CRITICAL, f"blocked: {reason}")
    
    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Read recent entries from the audit log."""
        entries = []
        try:
            if os.path.exists(self.audit_log_path):
                with open(self.audit_log_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.warning(f"Failed to read audit log: {e}")
        
        return entries
    
    def clear_audit_log(self) -> None:
        """Clear the audit log file."""
        try:
            with open(self.audit_log_path, "w") as f:
                f.write("")
            self.logger.info("🧹 Audit log cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear audit log: {e}")


def create_confirmation_callback(use_tts: bool = False) -> Callable[[str, RiskLevel], bool]:
    """
    Create a confirmation callback for high-risk operations.
    
    This can be customized to use TTS for voice prompts.
    """
    def confirm(message: str, risk_level: RiskLevel) -> bool:
        print(f"\n{'='*50}")
        print(f"⚠️  CONFIRMATION REQUIRED ({risk_level.value.upper()} RISK)")
        print(f"{'='*50}")
        print(message)
        print(f"{'='*50}")
        
        if use_tts:
            try:
                import subprocess
                subprocess.run(
                    ["say", f"Warning: {risk_level.value} risk operation detected. Please confirm."],
                    capture_output=True
                )
            except Exception:
                pass
        
        response = input("Allow this operation? (yes/no): ").strip().lower()
        return response in ("yes", "y")
    
    return confirm
