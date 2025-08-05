"""Input validation and sanitization utilities.

This module provides comprehensive validation for various inputs
to the formalization pipeline, ensuring data integrity and security.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from ..core.exceptions import ValidationError
from .logging_config import setup_logger


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    sanitized_input: Optional[Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """Comprehensive input validator for the formalization pipeline.
    
    This class provides validation for LaTeX content, file paths,
    configuration values, and other inputs to ensure security and
    data integrity.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """Initialize input validator.
        
        Args:
            validation_level: Strictness level for validation
        """
        self.validation_level = validation_level
        self.logger = setup_logger(__name__)
        
        # Dangerous patterns to check for
        self.dangerous_patterns = [
            # Command injection patterns
            r'[;&|`$()]',
            r'\\system\{',
            r'\\input\{',
            r'\\include\{',
            # Path traversal patterns
            r'\.\./|\.\.\\',
            # Script injection patterns
            r'<script[^>]*>',
            r'javascript:',
            # SQL injection patterns (less relevant but good practice)
            r'union\s+select',
            r'drop\s+table',
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.dangerous_patterns]
        
        # Maximum content sizes
        self.max_sizes = {
            "latex_content": 10 * 1024 * 1024,  # 10MB
            "file_size": 50 * 1024 * 1024,      # 50MB
            "theorem_statement": 10000,          # 10K characters
            "proof_content": 100000,             # 100K characters
        }
    
    def validate_latex_content(self, content: str) -> ValidationResult:
        """Validate LaTeX mathematical content.
        
        Args:
            content: LaTeX content to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        try:
            # Check basic requirements
            if not content or not content.strip():
                errors.append("LaTeX content is empty")
                return ValidationResult(False, errors, warnings)
            
            # Check content size
            if len(content) > self.max_sizes["latex_content"]:
                errors.append(f"LaTeX content too large: {len(content)} bytes "
                            f"(max: {self.max_sizes['latex_content']})")
            
            # Check for dangerous patterns
            dangerous_matches = self._check_dangerous_patterns(content)
            if dangerous_matches:
                errors.extend([f"Potentially dangerous pattern detected: {pattern}" 
                             for pattern in dangerous_matches])
            
            # Validate LaTeX structure
            structure_issues = self._validate_latex_structure(content)
            warnings.extend(structure_issues)
            
            # Check for mathematical content
            if not self._has_mathematical_content(content):
                warnings.append("No mathematical content detected")
            
            # Sanitize content
            sanitized_content = self._sanitize_latex_content(content)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                sanitized_input=sanitized_content
            )
            
        except Exception as e:
            self.logger.error(f"LaTeX validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {e}"]
            )
    
    def validate_file_path(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate file path for security and accessibility.
        
        Args:
            file_path: File path to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        try:
            # Convert to Path object
            path = Path(file_path)
            
            # Check for path traversal attempts
            path_str = str(path)
            if '..' in path_str or path_str.startswith('/'):
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append("Path traversal detected in file path")
                else:
                    warnings.append("Potentially unsafe file path")
            
            # Check if file exists (for input files)
            if not path.exists():
                errors.append(f"File does not exist: {path}")
            else:
                # Check file size
                file_size = path.stat().st_size
                if file_size > self.max_sizes["file_size"]:
                    errors.append(f"File too large: {file_size} bytes "
                                f"(max: {self.max_sizes['file_size']})")
                
                # Check file permissions
                if not os.access(path, os.R_OK):
                    errors.append(f"File not readable: {path}")
            
            # Check file extension
            if path.suffix.lower() not in ['.tex', '.latex', '.txt', '.md']:
                warnings.append(f"Unusual file extension: {path.suffix}")
            
            # Resolve absolute path for sanitization
            sanitized_path = path.resolve()
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                sanitized_input=sanitized_path
            )
            
        except Exception as e:
            self.logger.error(f"File path validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Path validation error: {e}"]
            )
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        sanitized_config = config_dict.copy()
        
        try:
            # Required fields
            required_fields = []  # Can be customized based on requirements
            
            for field in required_fields:
                if field not in config_dict:
                    errors.append(f"Required configuration field missing: {field}")
            
            # Validate specific configuration values
            if 'model' in config_dict:
                model_validation = self._validate_model_config(config_dict['model'])
                errors.extend(model_validation.errors)
                warnings.extend(model_validation.warnings)
            
            if 'timeout' in config_dict:
                timeout = config_dict['timeout']
                if not isinstance(timeout, int) or timeout <= 0:
                    errors.append("Timeout must be a positive integer")
                elif timeout > 300:  # 5 minutes
                    warnings.append("Very high timeout value specified")
            
            if 'max_tokens' in config_dict:
                max_tokens = config_dict['max_tokens']
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    errors.append("max_tokens must be a positive integer")
                elif max_tokens > 100000:
                    warnings.append("Very high max_tokens value specified")
            
            # Sanitize API keys and sensitive data
            if 'api_key' in sanitized_config:
                if sanitized_config['api_key']:
                    # Validate API key format but don't log it
                    api_key = sanitized_config['api_key']
                    if len(api_key) < 10:
                        errors.append("API key appears to be too short")
                    # Replace with placeholder for logging
                    sanitized_config['api_key'] = "[REDACTED]"
                else:
                    warnings.append("API key is empty")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                sanitized_input=sanitized_config
            )
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Configuration validation error: {e}"]
            )
    
    def validate_theorem_statement(self, statement: str) -> ValidationResult:
        """Validate a mathematical theorem statement.
        
        Args:
            statement: Theorem statement to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        try:
            if not statement or not statement.strip():
                errors.append("Theorem statement is empty")
                return ValidationResult(False, errors, warnings)
            
            # Check length
            if len(statement) > self.max_sizes["theorem_statement"]:
                errors.append(f"Theorem statement too long: {len(statement)} characters")
            
            # Check for dangerous patterns
            dangerous_matches = self._check_dangerous_patterns(statement)
            if dangerous_matches:
                errors.extend([f"Potentially dangerous pattern: {pattern}" 
                             for pattern in dangerous_matches])
            
            # Check for mathematical content
            if not self._has_mathematical_content(statement):
                warnings.append("Statement may not contain mathematical content")
            
            # Basic syntax validation
            if statement.count('{') != statement.count('}'):
                warnings.append("Unbalanced braces in statement")
            
            if statement.count('(') != statement.count(')'):
                warnings.append("Unbalanced parentheses in statement")
            
            sanitized_statement = self._sanitize_latex_content(statement)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                sanitized_input=sanitized_statement
            )
            
        except Exception as e:
            self.logger.error(f"Theorem statement validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Statement validation error: {e}"]
            )
    
    def _check_dangerous_patterns(self, content: str) -> List[str]:
        """Check content for dangerous patterns.
        
        Args:
            content: Content to check
            
        Returns:
            List of detected dangerous patterns
        """
        matches = []
        
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                matches.append(pattern.pattern)
        
        return matches
    
    def _validate_latex_structure(self, content: str) -> List[str]:
        """Validate LaTeX document structure.
        
        Args:
            content: LaTeX content
            
        Returns:
            List of structural issues
        """
        issues = []
        
        # Check for balanced environments
        environments = re.findall(r'\\begin\{([^}]+)\}', content)
        end_environments = re.findall(r'\\end\{([^}]+)\}', content)
        
        for env in environments:
            if environments.count(env) != end_environments.count(env):
                issues.append(f"Unbalanced environment: {env}")
        
        # Check for common LaTeX issues
        if '$$' in content and content.count('$$') % 2 != 0:
            issues.append("Unbalanced display math delimiters ($$)")
        
        if '$' in content and content.count('$') % 2 != 0:
            issues.append("Unbalanced inline math delimiters ($)")
        
        return issues
    
    def _has_mathematical_content(self, content: str) -> bool:
        """Check if content contains mathematical expressions.
        
        Args:
            content: Content to check
            
        Returns:
            True if mathematical content is detected
        """
        math_indicators = [
            r'\$[^$]+\$',  # Inline math
            r'\$\$[^$]+\$\$',  # Display math
            r'\\begin\{(equation|align|theorem|lemma|proof)\}',
            r'\\(sum|int|frac|sqrt|alpha|beta|gamma)',
            r'[+\-*/=<>≤≥∑∫∏∀∃]',
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _sanitize_latex_content(self, content: str) -> str:
        """Sanitize LaTeX content by removing/escaping dangerous elements.
        
        Args:
            content: LaTeX content to sanitize
            
        Returns:
            Sanitized content
        """
        sanitized = content
        
        # Remove potentially dangerous commands
        dangerous_commands = [
            r'\\input\{[^}]*\}',
            r'\\include\{[^}]*\}',
            r'\\system\{[^}]*\}',
            r'\\write18\{[^}]*\}',
        ]
        
        for pattern in dangerous_commands:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = sanitized.strip()
        
        return sanitized
    
    def _validate_model_config(self, model_config: Any) -> ValidationResult:
        """Validate model configuration.
        
        Args:
            model_config: Model configuration to validate
            
        Returns:
            ValidationResult for model configuration
        """
        errors = []
        warnings = []
        
        if isinstance(model_config, str):
            # Simple model name string
            if not model_config.strip():
                errors.append("Model name is empty")
            elif len(model_config) > 100:
                errors.append("Model name is too long")
        elif isinstance(model_config, dict):
            # Model configuration dictionary
            if 'name' not in model_config:
                errors.append("Model configuration missing 'name' field")
            
            if 'temperature' in model_config:
                temp = model_config['temperature']
                if not isinstance(temp, (int, float)):
                    errors.append("Temperature must be a number")
                elif temp < 0 or temp > 2:
                    errors.append("Temperature must be between 0 and 2")
        else:
            errors.append("Invalid model configuration format")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )