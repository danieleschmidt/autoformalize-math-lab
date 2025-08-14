"""Advanced security module for mathematical formalization pipeline.

This module provides comprehensive security features including:
- Input sanitization and validation
- Code injection prevention
- Rate limiting and abuse protection
- Audit logging and compliance
- Secure API key management
"""

import hashlib
import hmac
import time
import re
import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import secrets
from functools import wraps

from ..utils.logging_config import setup_logger


class SecurityLevel(Enum):
    """Security levels for different environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"


class ThreatType(Enum):
    """Types of security threats."""
    CODE_INJECTION = "code_injection"
    LATEX_INJECTION = "latex_injection"
    PATH_TRAVERSAL = "path_traversal"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MALICIOUS_PATTERN = "malicious_pattern"
    UNSAFE_API_USAGE = "unsafe_api_usage"


@dataclass
class SecurityEvent:
    """Represents a security event or threat."""
    event_id: str
    threat_type: ThreatType
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: float
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    payload: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    security_level: SecurityLevel = SecurityLevel.PRODUCTION
    enable_input_sanitization: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    max_input_length: int = 100000  # Max LaTeX input length
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_requests: int = 100  # Per hour
    rate_limit_window: int = 3600  # Seconds
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'\\input\{[^}]*\}',  # LaTeX file inclusion
        r'\\include\{[^}]*\}',
        r'\\write\d+\{[^}]*\}',  # LaTeX write commands
        r'\\openout\d+',
        r'\\closeout\d+',
        r'<script[^>]*>',  # HTML script tags
        r'javascript:',
        r'eval\s*\(',  # JavaScript eval
        r'exec\s*\(',  # Python exec
        r'system\s*\(',  # System calls
        r'subprocess\.',
        r'os\.(system|popen|spawn)',
        r'__import__\s*\(',
    ])
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.tex', '.latex', '.ltx', '.sty', '.cls'
    })


class SecurityValidator:
    """Comprehensive security validator for mathematical formalization."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security validator.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self.logger = setup_logger(__name__)
        self.security_events: List[SecurityEvent] = []
        self.rate_limit_tracking: Dict[str, List[float]] = {}
        
        # Compile regex patterns for efficiency
        self.blocked_pattern_regex = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.config.blocked_patterns
        ]
        
        self.logger.info(f"Security validator initialized with {self.config.security_level.value} level")
    
    def validate_latex_input(self, latex_content: str, source_ip: Optional[str] = None) -> bool:
        """Validate LaTeX input for security threats.
        
        Args:
            latex_content: LaTeX content to validate
            source_ip: Source IP address for tracking
            
        Returns:
            True if input is safe, False otherwise
            
        Raises:
            SecurityError: If critical security threat detected
        """
        # Check input length
        if len(latex_content) > self.config.max_input_length:
            self._log_security_event(
                ThreatType.MALICIOUS_PATTERN,
                "HIGH",
                f"Input length exceeds limit: {len(latex_content)}",
                source_ip=source_ip
            )
            return False
        
        # Check for blocked patterns
        for pattern_regex in self.blocked_pattern_regex:
            if pattern_regex.search(latex_content):
                self._log_security_event(
                    ThreatType.LATEX_INJECTION,
                    "HIGH",
                    f"Blocked pattern detected: {pattern_regex.pattern}",
                    source_ip=source_ip,
                    payload=latex_content[:200]  # Log first 200 chars
                )
                return False
        
        # Check for suspicious command sequences
        if self._detect_command_injection(latex_content):
            self._log_security_event(
                ThreatType.CODE_INJECTION,
                "CRITICAL",
                "Potential command injection detected",
                source_ip=source_ip,
                payload=latex_content[:200]
            )
            return False
        
        # Check for path traversal attempts
        if self._detect_path_traversal(latex_content):
            self._log_security_event(
                ThreatType.PATH_TRAVERSAL,
                "HIGH",
                "Path traversal attempt detected",
                source_ip=source_ip,
                payload=latex_content[:200]
            )
            return False
        
        self.logger.debug("LaTeX input validation passed")
        return True
    
    def sanitize_latex_input(self, latex_content: str) -> str:
        """Sanitize LaTeX input by removing or escaping dangerous content.
        
        Args:
            latex_content: Raw LaTeX content
            
        Returns:
            Sanitized LaTeX content
        """
        if not self.config.enable_input_sanitization:
            return latex_content
        
        sanitized = latex_content
        
        # Remove potentially dangerous commands
        dangerous_commands = [
            r'\\input\{[^}]*\}',
            r'\\include\{[^}]*\}',
            r'\\write\d+\{[^}]*\}',
            r'\\openout\d+',
            r'\\closeout\d+',
        ]
        
        for pattern in dangerous_commands:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Escape special characters that could be used maliciously
        sanitized = sanitized.replace('\\catcode', '\\textbackslash catcode')
        sanitized = sanitized.replace('\\def', '\\textbackslash def')
        
        # Limit nesting depth to prevent DoS
        sanitized = self._limit_nesting_depth(sanitized)
        
        if sanitized != latex_content:
            self.logger.info("LaTeX input was sanitized")
        
        return sanitized
    
    def validate_file_upload(self, file_path: Path, content: bytes) -> bool:
        """Validate uploaded file for security.
        
        Args:
            file_path: Path to uploaded file
            content: File content as bytes
            
        Returns:
            True if file is safe, False otherwise
        """
        # Check file extension
        if file_path.suffix.lower() not in self.config.allowed_file_extensions:
            self._log_security_event(
                ThreatType.MALICIOUS_PATTERN,
                "MEDIUM",
                f"Disallowed file extension: {file_path.suffix}",
                metadata={'filename': str(file_path)}
            )
            return False
        
        # Check file size
        if len(content) > self.config.max_file_size:
            self._log_security_event(
                ThreatType.MALICIOUS_PATTERN,
                "MEDIUM",
                f"File size exceeds limit: {len(content)} bytes",
                metadata={'filename': str(file_path)}
            )
            return False
        
        # Check for binary content in text files
        if self._contains_binary_content(content):
            self._log_security_event(
                ThreatType.MALICIOUS_PATTERN,
                "HIGH",
                "Binary content detected in text file",
                metadata={'filename': str(file_path)}
            )
            return False
        
        # Validate content as LaTeX
        try:
            text_content = content.decode('utf-8', errors='strict')
            return self.validate_latex_input(text_content)
        except UnicodeDecodeError:
            self._log_security_event(
                ThreatType.MALICIOUS_PATTERN,
                "MEDIUM",
                "Invalid UTF-8 encoding in file",
                metadata={'filename': str(file_path)}
            )
            return False
    
    def check_rate_limit(self, identifier: str, source_ip: Optional[str] = None) -> bool:
        """Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier for rate limiting (e.g., user ID, IP)
            source_ip: Source IP address
            
        Returns:
            True if within limits, False if rate limit exceeded
        """
        if not self.config.enable_rate_limiting:
            return True
        
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        # Clean old entries
        if identifier in self.rate_limit_tracking:
            self.rate_limit_tracking[identifier] = [
                timestamp for timestamp in self.rate_limit_tracking[identifier]
                if timestamp > window_start
            ]
        else:
            self.rate_limit_tracking[identifier] = []
        
        # Check if limit exceeded
        if len(self.rate_limit_tracking[identifier]) >= self.config.rate_limit_requests:
            self._log_security_event(
                ThreatType.RATE_LIMIT_EXCEEDED,
                "MEDIUM",
                f"Rate limit exceeded for {identifier}",
                source_ip=source_ip,
                metadata={'requests_count': len(self.rate_limit_tracking[identifier])}
            )
            return False
        
        # Record this request
        self.rate_limit_tracking[identifier].append(current_time)
        return True
    
    def validate_api_key(self, api_key: str, expected_prefix: str = "sk-") -> bool:
        """Validate API key format and security.
        
        Args:
            api_key: API key to validate
            expected_prefix: Expected key prefix
            
        Returns:
            True if key appears valid, False otherwise
        """
        if not api_key:
            return False
        
        # Check prefix
        if not api_key.startswith(expected_prefix):
            self._log_security_event(
                ThreatType.UNSAFE_API_USAGE,
                "LOW",
                "API key with invalid prefix detected"
            )
            return False
        
        # Check length (typical API keys are 40+ characters)
        if len(api_key) < 20:
            self._log_security_event(
                ThreatType.UNSAFE_API_USAGE,
                "MEDIUM",
                "Suspiciously short API key detected"
            )
            return False
        
        # Check for obvious test/dummy keys
        dummy_patterns = ['test', 'dummy', 'fake', 'example', '123456']
        api_key_lower = api_key.lower()
        
        for pattern in dummy_patterns:
            if pattern in api_key_lower:
                self._log_security_event(
                    ThreatType.UNSAFE_API_USAGE,
                    "HIGH",
                    f"Test/dummy API key pattern detected: {pattern}"
                )
                return False
        
        return True
    
    def _detect_command_injection(self, content: str) -> bool:
        """Detect potential command injection attempts."""
        # Look for command execution patterns
        injection_patterns = [
            r'`[^`]*`',  # Backticks
            r'\$\([^)]*\)',  # Command substitution
            r';[\s]*[a-zA-Z]',  # Command chaining
            r'\|[\s]*[a-zA-Z]',  # Pipes
            r'&&[\s]*[a-zA-Z]',  # AND chaining
            r'\|\|[\s]*[a-zA-Z]',  # OR chaining
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _detect_path_traversal(self, content: str) -> bool:
        """Detect path traversal attempts."""
        traversal_patterns = [
            r'\.\./.*\.\.',  # Classic path traversal
            r'\.\.\\.*\.\.',  # Windows style
            r'/etc/passwd',  # Common target
            r'/proc/',  # Linux proc filesystem
            r'C:\\Windows',  # Windows system
            r'%2e%2e',  # URL encoded dots
            r'\.\.%2f',  # Mixed encoding
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _limit_nesting_depth(self, latex_content: str, max_depth: int = 10) -> str:
        """Limit nesting depth to prevent DoS attacks."""
        # Count nesting depth of common LaTeX constructs
        depth_patterns = [
            (r'\{', r'\}'),  # Braces
            (r'\\begin\{[^}]+\}', r'\\end\{[^}]+\}'),  # Environments
        ]
        
        for open_pattern, close_pattern in depth_patterns:
            latex_content = self._limit_pattern_depth(
                latex_content, open_pattern, close_pattern, max_depth
            )
        
        return latex_content
    
    def _limit_pattern_depth(
        self, 
        content: str, 
        open_pattern: str, 
        close_pattern: str, 
        max_depth: int
    ) -> str:
        """Limit depth of specific pattern to prevent DoS."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated parsing
        open_count = len(re.findall(open_pattern, content))
        close_count = len(re.findall(close_pattern, content))
        
        if open_count > max_depth * 2:  # Simple heuristic
            self.logger.warning(f"Limiting nested pattern depth: {open_pattern}")
            # Remove excess patterns (simplified)
            content = re.sub(open_pattern, '', content)[:(len(content)//2)]
        
        return content
    
    def _contains_binary_content(self, content: bytes) -> bool:
        """Check if content contains binary data."""
        try:
            # Try to decode as UTF-8
            content.decode('utf-8')
            
            # Check for null bytes
            if b'\x00' in content:
                return True
            
            # Check for high percentage of non-printable characters
            text_content = content.decode('utf-8', errors='ignore')
            printable_chars = sum(1 for c in text_content if c.isprintable() or c.isspace())
            
            if len(text_content) > 0:
                printable_ratio = printable_chars / len(text_content)
                return printable_ratio < 0.8  # Less than 80% printable
            
            return False
            
        except UnicodeDecodeError:
            return True
    
    def _log_security_event(
        self,
        threat_type: ThreatType,
        severity: str,
        message: str,
        source_ip: Optional[str] = None,
        payload: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event."""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            threat_type=threat_type,
            severity=severity,
            timestamp=time.time(),
            source_ip=source_ip,
            payload=payload,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Log to standard logger
        log_level = {
            'LOW': logging.INFO,
            'MEDIUM': logging.WARNING,
            'HIGH': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(
            log_level,
            f"SECURITY EVENT [{threat_type.value}] {message} "
            f"(ID: {event.event_id[:8]}, IP: {source_ip or 'unknown'})"
        )
        
        # In production, you might also send to SIEM or security monitoring
        if self.config.security_level == SecurityLevel.PRODUCTION and severity in ['HIGH', 'CRITICAL']:
            self._send_security_alert(event)
    
    def _send_security_alert(self, event: SecurityEvent) -> None:
        """Send security alert to monitoring system."""
        # Placeholder for security alert system
        # In production, this would integrate with SIEM, Slack, PagerDuty, etc.
        self.logger.critical(f"SECURITY ALERT: {event.threat_type.value} - {event.severity}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security events summary."""
        if not self.security_events:
            return {
                'total_events': 0,
                'by_threat_type': {},
                'by_severity': {},
                'recent_events': []
            }
        
        # Count by threat type
        by_threat = {}
        for event in self.security_events:
            threat_type = event.threat_type.value
            by_threat[threat_type] = by_threat.get(threat_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for event in self.security_events:
            severity = event.severity
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Recent events (last 10)
        recent_events = [
            {
                'event_id': event.event_id[:8],
                'threat_type': event.threat_type.value,
                'severity': event.severity,
                'timestamp': event.timestamp,
                'source_ip': event.source_ip
            }
            for event in sorted(self.security_events, key=lambda e: e.timestamp, reverse=True)[:10]
        ]
        
        return {
            'total_events': len(self.security_events),
            'by_threat_type': by_threat,
            'by_severity': by_severity,
            'recent_events': recent_events,
            'rate_limit_tracking': {
                k: len(v) for k, v in self.rate_limit_tracking.items()
            }
        }
    
    def export_security_log(self, file_path: Path) -> None:
        """Export security events to file."""
        log_data = {
            'export_timestamp': time.time(),
            'security_config': {
                'security_level': self.config.security_level.value,
                'max_input_length': self.config.max_input_length,
                'rate_limit_requests': self.config.rate_limit_requests,
            },
            'events': [
                {
                    'event_id': event.event_id,
                    'threat_type': event.threat_type.value,
                    'severity': event.severity,
                    'timestamp': event.timestamp,
                    'source_ip': event.source_ip,
                    'metadata': event.metadata
                }
                for event in self.security_events
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Security log exported to {file_path}")


def security_required(security_level: SecurityLevel = SecurityLevel.PRODUCTION):
    """Decorator to require security validation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security validator from args/kwargs or create one
            validator = None
            
            # Look for security validator in instance
            if args and hasattr(args[0], 'security_validator'):
                validator = args[0].security_validator
            
            if not validator:
                validator = SecurityValidator(SecurityConfig(security_level=security_level))
            
            # Basic security checks
            if 'latex_content' in kwargs:
                if not validator.validate_latex_input(kwargs['latex_content']):
                    raise SecurityError("LaTeX input failed security validation")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class SecurityError(Exception):
    """Security-related error."""
    pass