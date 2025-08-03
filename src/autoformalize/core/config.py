"""Configuration management for the formalization pipeline.

This module provides configuration classes and utilities for managing
pipeline settings, API keys, and system preferences.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from pydantic import BaseModel, Field
    from pydantic_settings import BaseSettings
    HAS_PYDANTIC = True
except ImportError:
    # Fallback for basic configuration
    BaseModel = object
    BaseSettings = object
    Field = lambda default=None, **kwargs: default
    HAS_PYDANTIC = False

from .exceptions import ConfigurationError


class LogLevel(Enum):
    """Available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.api_key:
            # Try to get from environment
            if self.name.startswith("gpt") or self.name.startswith("claude"):
                self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")


@dataclass
class VerificationConfig:
    """Configuration for proof verification."""
    enabled: bool = True
    timeout: int = 30
    max_memory_mb: int = 1024
    lean_executable: Optional[str] = None
    isabelle_executable: Optional[str] = None
    coq_executable: Optional[str] = None
    
    def get_executable(self, system: str) -> Optional[str]:
        """Get executable path for proof assistant system."""
        if system == "lean4":
            return self.lean_executable or "lean"
        elif system == "isabelle":
            return self.isabelle_executable or "isabelle"
        elif system == "coq":
            return self.coq_executable or "coqc"
        return None


@dataclass
class ParsingConfig:
    """Configuration for LaTeX parsing."""
    extract_theorems: bool = True
    extract_definitions: bool = True
    extract_proofs: bool = True
    extract_lemmas: bool = True
    ignore_comments: bool = True
    preserve_formatting: bool = False
    custom_environments: List[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration for formal code generation."""
    use_mathlib: bool = True
    include_comments: bool = True
    generate_sorry: bool = False  # Generate 'sorry' for incomplete proofs
    max_proof_length: int = 1000
    custom_templates: Dict[str, str] = field(default_factory=dict)
    prompt_templates_dir: Optional[Path] = None


@dataclass
class FormalizationConfig:
    """Main configuration class for the formalization pipeline.
    
    This class aggregates all configuration settings for the pipeline
    and provides methods for loading from files and environment variables.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # General settings
    log_level: LogLevel = LogLevel.INFO
    output_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    max_workers: int = 4
    enable_metrics: bool = True
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'FormalizationConfig':
        """Load configuration from a file.
        
        Supports JSON, YAML, and TOML formats.
        """
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            import json
            
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        data = yaml.safe_load(f)
                except ImportError:
                    raise ConfigurationError("PyYAML required for YAML configuration files")
            elif config_path.suffix.lower() == '.toml':
                try:
                    import tomli
                    with open(config_path, 'rb') as f:
                        data = tomli.load(f)
                except ImportError:
                    raise ConfigurationError("tomli required for TOML configuration files")
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
            
            return cls.from_dict(data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormalizationConfig':
        """Create configuration from dictionary."""
        try:
            # Extract nested configurations
            model_data = data.get('model', {})
            verification_data = data.get('verification', {})
            parsing_data = data.get('parsing', {})
            generation_data = data.get('generation', {})
            
            # Create nested config objects
            model_config = ModelConfig(**model_data)
            verification_config = VerificationConfig(**verification_data)
            parsing_config = ParsingConfig(**parsing_data)
            generation_config = GenerationConfig(**generation_data)
            
            # Create main config
            main_data = {k: v for k, v in data.items() 
                        if k not in ['model', 'verification', 'parsing', 'generation']}
            
            # Handle enum conversion
            if 'log_level' in main_data:
                main_data['log_level'] = LogLevel(main_data['log_level'])
            
            # Handle Path conversion
            for path_field in ['output_dir', 'cache_dir']:
                if path_field in main_data and main_data[path_field]:
                    main_data[path_field] = Path(main_data[path_field])
            
            return cls(
                model=model_config,
                verification=verification_config,
                parsing=parsing_config,
                generation=generation_config,
                **main_data
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration from dict: {e}")
    
    @classmethod
    def from_env(cls) -> 'FormalizationConfig':
        """Create configuration from environment variables.
        
        Environment variables should be prefixed with 'AUTOFORMALIZE_'.
        """
        env_data = {}
        
        # Extract environment variables
        for key, value in os.environ.items():
            if key.startswith('AUTOFORMALIZE_'):
                config_key = key[13:].lower()  # Remove prefix and lowercase
                env_data[config_key] = value
        
        # Convert string values to appropriate types
        type_conversions = {
            'max_workers': int,
            'enable_metrics': lambda x: x.lower() in ('true', '1', 'yes'),
            'timeout': int,
            'max_tokens': int,
            'temperature': float,
            'max_retries': int,
        }
        
        for key, converter in type_conversions.items():
            if key in env_data:
                try:
                    env_data[key] = converter(env_data[key])
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Invalid value for {key}: {env_data[key]}")
        
        return cls.from_dict(env_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            'model': {
                'name': self.model.name,
                'max_tokens': self.model.max_tokens,
                'temperature': self.model.temperature,
                'timeout': self.model.timeout,
                'max_retries': self.model.max_retries,
            },
            'verification': {
                'enabled': self.verification.enabled,
                'timeout': self.verification.timeout,
                'max_memory_mb': self.verification.max_memory_mb,
            },
            'parsing': {
                'extract_theorems': self.parsing.extract_theorems,
                'extract_definitions': self.parsing.extract_definitions,
                'extract_proofs': self.parsing.extract_proofs,
                'extract_lemmas': self.parsing.extract_lemmas,
                'ignore_comments': self.parsing.ignore_comments,
                'preserve_formatting': self.parsing.preserve_formatting,
                'custom_environments': self.parsing.custom_environments,
            },
            'generation': {
                'use_mathlib': self.generation.use_mathlib,
                'include_comments': self.generation.include_comments,
                'generate_sorry': self.generation.generate_sorry,
                'max_proof_length': self.generation.max_proof_length,
                'custom_templates': self.generation.custom_templates,
            },
            'log_level': self.log_level.value,
            'max_workers': self.max_workers,
            'enable_metrics': self.enable_metrics,
        }
        
        # Add optional paths if set
        if self.output_dir:
            result['output_dir'] = str(self.output_dir)
        if self.cache_dir:
            result['cache_dir'] = str(self.cache_dir)
            
        return result
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to a file."""
        try:
            import json
            
            data = self.to_dict()
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    with open(config_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False)
                except ImportError:
                    raise ConfigurationError("PyYAML required for YAML configuration files")
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {config_path}: {e}")
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate model settings
        if self.model.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        if not 0 <= self.model.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        if self.model.timeout <= 0:
            errors.append("model timeout must be positive")
        
        # Validate verification settings
        if self.verification.timeout <= 0:
            errors.append("verification timeout must be positive")
        if self.verification.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        
        # Validate general settings
        if self.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")


# Default configuration instance
default_config = FormalizationConfig()
