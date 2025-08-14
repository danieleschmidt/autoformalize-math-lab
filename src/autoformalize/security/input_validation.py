"""Advanced input validation and sanitization for mathematical content.

This module provides comprehensive validation for LaTeX inputs, mathematical
expressions, and user-provided data to ensure security and correctness.
"""

import re
import html
import logging
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging_config import setup_logger


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    sanitized_content: Optional[str] = None
    risk_score: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class LaTeXValidator:
    """Validates and sanitizes LaTeX mathematical content."""
    
    # Allowed LaTeX commands for mathematical content
    ALLOWED_MATH_COMMANDS = {
        # Basic math
        'frac', 'sqrt', 'sum', 'prod', 'int', 'lim', 'infty', 'partial',
        # Greek letters
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
        'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma',
        'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega',
        # Math operators
        'cdot', 'times', 'div', 'pm', 'mp', 'cap', 'cup', 'subset', 'supset',
        'subseteq', 'supseteq', 'in', 'notin', 'exists', 'forall', 'neg',
        'land', 'lor', 'implies', 'iff', 'equiv', 'approx', 'neq', 'leq', 'geq',
        # Structures
        'left', 'right', 'begin', 'end', 'text', 'mathrm', 'mathbb', 'mathcal',
        'mathfrak', 'boldsymbol', 'bm',
        # Environments
        'equation', 'align', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
        'cases', 'split',
    }
    
    # Allowed mathematical environments
    ALLOWED_MATH_ENVIRONMENTS = {
        'theorem', 'lemma', 'corollary', 'proposition', 'definition', 'proof',
        'example', 'remark', 'note', 'claim', 'fact', 'observation',
        'equation', 'align', 'gather', 'multline', 'split', 'aligned',
        'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'cases',
        'array', 'tabular',
    }
    
    # Potentially dangerous commands (banned)
    DANGEROUS_COMMANDS = {
        'input', 'include', 'write', 'openout', 'closeout', 'read', 'openin',
        'closein', 'immediate', 'special', 'pdfshellescape', 'write18',
        'catcode', 'def', 'gdef', 'edef', 'xdef', 'let', 'futurelet',
        'expandafter', 'noexpand', 'csname', 'endcsname', 'string', 'meaning',
        'jobname', 'InputIfFileExists', 'RequirePackage', 'usepackage',
        'documentclass', 'LoadClass', 'PassOptionsToClass', 'PassOptionsToPackage',
    }
    
    # Maximum content size limits
    MAX_CONTENT_SIZE = 1024 * 1024  # 1MB
    MAX_NESTING_DEPTH = 20
    MAX_COMMAND_COUNT = 10000
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for validation."""
        # Pattern for LaTeX commands
        self.command_pattern = re.compile(r'\\([a-zA-Z]+)(?:\*)?(?:\[[^\]]*\])?(?:\{[^}]*\})*')
        
        # Pattern for environments
        self.env_pattern = re.compile(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', re.DOTALL)
        
        # Pattern for potentially dangerous sequences
        self.dangerous_pattern = re.compile(
            r'\\(?:' + '|'.join(self.DANGEROUS_COMMANDS) + r')',
            re.IGNORECASE
        )
        
        # Pattern for suspicious character sequences
        self.suspicious_pattern = re.compile(
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]|'  # Control characters
            r'<script|javascript:|data:text/html|vbscript:|'  # Web injection
            r'\\catcode|\\def\\|\\gdef\\|\\let\\'  # TeX programming
        )
        
        # Pattern for excessive repetition
        self.repetition_pattern = re.compile(r'(.)\1{100,}')
    
    def validate(self, content: str, strict: bool = True) -> ValidationResult:
        """Validate LaTeX mathematical content."""
        result = ValidationResult()
        
        try:
            # Basic size and format checks
            if not self._check_basic_format(content, result):
                return result
            
            # Check for dangerous commands
            if not self._check_dangerous_commands(content, result):
                return result
            
            # Validate LaTeX commands
            if not self._validate_commands(content, result, strict):
                return result
            
            # Validate environments
            if not self._validate_environments(content, result, strict):
                return result
            
            # Check nesting depth
            if not self._check_nesting_depth(content, result):
                return result
            
            # Check for suspicious patterns
            if not self._check_suspicious_patterns(content, result):
                return result
            
            # Sanitize content
            result.sanitized_content = self._sanitize_content(content)
            
            # Calculate risk score
            result.risk_score = self._calculate_risk_score(content, result)
            
            if result.errors:
                result.is_valid = False
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            result.is_valid = False
            result.errors.append(f"Validation system error: {str(e)}")
        
        return result
    
    def _check_basic_format(self, content: str, result: ValidationResult) -> bool:
        """Check basic format constraints."""
        if not isinstance(content, str):
            result.errors.append("Content must be a string")
            return False
        
        if len(content) > self.MAX_CONTENT_SIZE:
            result.errors.append(f"Content too large: {len(content)} > {self.MAX_CONTENT_SIZE}")
            return False
        
        if len(content.strip()) == 0:
            result.warnings.append("Content is empty")
            return True
        
        # Check for excessive repetition
        if self.repetition_pattern.search(content):
            result.errors.append("Excessive character repetition detected")
            return False
        
        return True
    
    def _check_dangerous_commands(self, content: str, result: ValidationResult) -> bool:
        """Check for dangerous LaTeX commands."""
        matches = self.dangerous_pattern.findall(content)
        if matches:
            result.errors.append(f"Dangerous commands detected: {', '.join(set(matches))}")
            return False
        return True
    
    def _validate_commands(self, content: str, result: ValidationResult, strict: bool) -> bool:
        """Validate LaTeX commands."""
        commands = self.command_pattern.findall(content)
        
        if len(commands) > self.MAX_COMMAND_COUNT:
            result.errors.append(f"Too many commands: {len(commands)} > {self.MAX_COMMAND_COUNT}")
            return False
        
        if strict:
            unknown_commands = set(commands) - self.ALLOWED_MATH_COMMANDS
            if unknown_commands:
                result.warnings.append(f"Unknown commands: {', '.join(unknown_commands)}")
        
        return True
    
    def _validate_environments(self, content: str, result: ValidationResult, strict: bool) -> bool:
        """Validate LaTeX environments."""
        environments = re.findall(r'\\begin\{([^}]+)\}', content)
        
        if strict:
            unknown_envs = set(environments) - self.ALLOWED_MATH_ENVIRONMENTS
            if unknown_envs:
                result.warnings.append(f"Unknown environments: {', '.join(unknown_envs)}")
        
        # Check for properly closed environments
        for env_match in self.env_pattern.finditer(content):
            env_name = env_match.group(1)
            env_content = env_match.group(2)
            
            # Check for nested environments of the same type
            nested_begins = len(re.findall(rf'\\begin\{{{re.escape(env_name)}\}}', env_content))
            if nested_begins > 0:
                result.warnings.append(f"Nested '{env_name}' environments detected")
        
        return True
    
    def _check_nesting_depth(self, content: str, result: ValidationResult) -> bool:
        """Check nesting depth of braces and environments."""
        max_depth = 0
        current_depth = 0
        
        for char in content:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        if max_depth > self.MAX_NESTING_DEPTH:
            result.errors.append(f"Nesting too deep: {max_depth} > {self.MAX_NESTING_DEPTH}")
            return False
        
        if current_depth != 0:
            result.errors.append("Unmatched braces detected")
            return False
        
        return True
    
    def _check_suspicious_patterns(self, content: str, result: ValidationResult) -> bool:
        """Check for suspicious patterns that might indicate attacks."""
        if self.suspicious_pattern.search(content):
            result.errors.append("Suspicious patterns detected")
            return False
        
        # Check for unusual Unicode characters
        unusual_chars = []
        for char in content:
            if ord(char) > 0x1F000:  # Beyond typical mathematical Unicode
                unusual_chars.append(char)
        
        if unusual_chars and len(set(unusual_chars)) > 10:
            result.warnings.append(f"Unusual Unicode characters: {len(set(unusual_chars))}")
        
        return True
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize LaTeX content by removing or escaping dangerous elements."""
        # Remove dangerous commands
        sanitized = self.dangerous_pattern.sub('', content)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', sanitized)
        
        return sanitized.strip()
    
    def _calculate_risk_score(self, content: str, result: ValidationResult) -> float:
        """Calculate risk score from 0.0 (safe) to 1.0 (dangerous)."""
        score = 0.0
        
        # Factor in errors and warnings
        score += len(result.errors) * 0.3
        score += len(result.warnings) * 0.1
        
        # Size-based risk
        if len(content) > 50000:
            score += 0.2
        elif len(content) > 10000:
            score += 0.1
        
        # Complexity-based risk
        command_count = len(self.command_pattern.findall(content))
        if command_count > 1000:
            score += 0.3
        elif command_count > 100:
            score += 0.1
        
        return min(score, 1.0)


class ContentSanitizer:
    """Sanitizes various types of mathematical content."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.latex_validator = LaTeXValidator()
    
    def sanitize_latex(self, content: str, strict: bool = True) -> ValidationResult:
        """Sanitize LaTeX mathematical content."""
        return self.latex_validator.validate(content, strict)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent directory traversal."""
        # Remove any path components
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        safe_chars = re.sub(r'[^\w\.-]', '_', filename)
        
        # Ensure reasonable length
        if len(safe_chars) > 200:
            name, ext = Path(safe_chars).stem[:190], Path(safe_chars).suffix
            safe_chars = f"{name}{ext}"
        
        return safe_chars
    
    def sanitize_text_input(self, text: str, max_length: int = 10000) -> str:
        """Sanitize general text input."""
        if not isinstance(text, str):
            return ""
        
        # Limit length
        text = text[:max_length]
        
        # HTML escape
        text = html.escape(text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        return text.strip()
    
    def generate_content_hash(self, content: str) -> str:
        """Generate a hash for content integrity checking."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, time_window: float = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}
        self.logger = setup_logger(__name__)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the identifier."""
        now = time.time()
        
        # Initialize if new identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Record this request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for the identifier."""
        if identifier not in self.requests:
            return self.max_requests
        
        now = time.time()
        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        return max(0, self.max_requests - len(recent_requests))


# Global instances
content_sanitizer = ContentSanitizer()
rate_limiter = RateLimiter()