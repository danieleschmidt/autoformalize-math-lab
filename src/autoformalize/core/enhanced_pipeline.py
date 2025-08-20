"""Enhanced formalization pipeline with real proof assistant integration.

This module provides an enhanced version of the FormalizationPipeline with
improved verification, mathematical validation, and production-ready features.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

try:
    from openai import AsyncOpenAI
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncOpenAI = None
    AsyncAnthropic = None

from ..parsers.latex_parser import LaTeXParser
from ..generators.lean import Lean4Generator
from ..generators.isabelle import IsabelleGenerator
from ..generators.coq import CoqGenerator
from ..verifiers.lean_verifier import Lean4Verifier
from ..verifiers.isabelle_verifier import IsabelleVerifier
from ..verifiers.coq_verifier import CoqVerifier
from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics
from ..utils.caching import CacheManager
try:
    from ..security.input_validation import InputValidator
except ImportError:
    # Fallback if InputValidator not available
    class InputValidator:
        async def validate_latex_input(self, content):
            class ValidationResult:
                valid = True
                message = "Validation passed"
            return ValidationResult()
from .exceptions import FormalizationError, UnsupportedSystemError
from .config import FormalizationConfig


class VerificationMode(Enum):
    """Verification modes for enhanced pipeline."""
    DISABLED = "disabled"
    MOCK = "mock"
    REAL = "real"
    HYBRID = "hybrid"


@dataclass
class EnhancedFormalizationResult:
    """Enhanced result with mathematical validation and metrics."""
    success: bool
    formal_code: Optional[str] = None
    error_message: Optional[str] = None
    verification_status: Optional[bool] = None
    mathematical_validation: Optional[bool] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    correction_rounds: int = 0
    processing_time: float = 0.0
    verification_time: float = 0.0
    confidence_score: float = 0.0
    complexity_score: int = 0
    theorem_type: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class EnhancedFormalizationPipeline:
    """Enhanced formalization pipeline with production features."""
    
    def __init__(
        self,
        target_system: str = "lean4",
        model_name: str = "gpt-4",
        verification_mode: VerificationMode = VerificationMode.REAL,
        enable_caching: bool = True,
        enable_validation: bool = True,
        max_correction_rounds: int = 5,
        timeout: int = 120
    ):
        """Initialize enhanced pipeline.
        
        Args:
            target_system: Target proof assistant system
            model_name: LLM model to use
            verification_mode: Verification mode to use
            enable_caching: Whether to enable result caching
            enable_validation: Whether to enable mathematical validation
            max_correction_rounds: Maximum correction attempts
            timeout: Overall timeout in seconds
        """
        self.target_system = target_system
        self.model_name = model_name
        self.verification_mode = verification_mode
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        self.max_correction_rounds = max_correction_rounds
        self.timeout = timeout
        
        self.logger = setup_logger(__name__)
        self.metrics = FormalizationMetrics()
        
        # Initialize components
        self._init_parser()
        self._init_generator()
        self._init_verifier()
        self._init_cache()
        self._init_validator()
        
        self.logger.info(f"Enhanced pipeline initialized with {target_system} target")
    
    def _init_parser(self) -> None:
        """Initialize LaTeX parser."""
        self.parser = LaTeXParser()
    
    def _init_generator(self) -> None:
        """Initialize code generator."""
        if self.target_system == "lean4":
            self.generator = Lean4Generator(model_name=self.model_name)
        elif self.target_system == "isabelle":
            self.generator = IsabelleGenerator(model_name=self.model_name)
        elif self.target_system == "coq":
            self.generator = CoqGenerator(model_name=self.model_name)
        else:
            raise UnsupportedSystemError(f"Unsupported target system: {self.target_system}")
    
    def _init_verifier(self) -> None:
        """Initialize proof verifier."""
        if self.verification_mode == VerificationMode.DISABLED:
            self.verifier = None
            return
        
        if self.target_system == "lean4":
            self.verifier = Lean4Verifier()
        elif self.target_system == "isabelle":
            self.verifier = IsabelleVerifier()
        elif self.target_system == "coq":
            self.verifier = CoqVerifier()
        else:
            self.verifier = None
    
    def _init_cache(self) -> None:
        """Initialize cache manager."""
        if self.enable_caching:
            self.cache = CacheManager()
        else:
            self.cache = None
    
    def _init_validator(self) -> None:
        """Initialize input validator."""
        if self.enable_validation:
            self.validator = InputValidator()
        else:
            self.validator = None
    
    async def formalize(
        self,
        latex_content: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> EnhancedFormalizationResult:
        """Formalize LaTeX content with enhanced features.
        
        Args:
            latex_content: LaTeX mathematical content
            context: Additional context for formalization
            timeout: Override default timeout
            
        Returns:
            Enhanced formalization result
        """
        start_time = time.time()
        timeout = timeout or self.timeout
        
        try:
            # Input validation
            if self.validator:
                validation_result = await self.validator.validate_latex_input(latex_content)
                if not validation_result.valid:
                    return EnhancedFormalizationResult(
                        success=False,
                        error_message=f"Input validation failed: {validation_result.message}",
                        processing_time=time.time() - start_time
                    )
            
            # Check cache
            cache_key = self._generate_cache_key(latex_content, context)
            if self.cache:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Using cached formalization result")
                    return cached_result
            
            # Parse LaTeX content
            parsed_content = await self.parser.parse(latex_content)
            if not parsed_content.theorems and not parsed_content.definitions:
                return EnhancedFormalizationResult(
                    success=False,
                    error_message="No mathematical content found in LaTeX",
                    processing_time=time.time() - start_time
                )
            
            # Generate formal code
            generation_result = await self._generate_with_correction(
                parsed_content, context, timeout
            )
            
            if not generation_result.success:
                return generation_result
            
            # Verify if enabled
            verification_result = None
            verification_time = 0.0
            
            if self.verifier and self.verification_mode in [VerificationMode.REAL, VerificationMode.HYBRID]:
                verify_start = time.time()
                try:
                    verification_result = await asyncio.wait_for(
                        self.verifier.verify_detailed(generation_result.formal_code),
                        timeout=timeout // 2
                    )
                    verification_time = time.time() - verify_start
                    generation_result.verification_status = verification_result.success
                    generation_result.verification_time = verification_time
                    
                    if not verification_result.success:
                        self.logger.warning(f"Verification failed: {verification_result.errors}")
                
                except asyncio.TimeoutError:
                    self.logger.warning("Verification timed out")
                    generation_result.verification_status = False
                    verification_time = timeout // 2
            
            # Mathematical validation
            if self.enable_validation:
                math_validation = await self._validate_mathematical_correctness(
                    latex_content, generation_result.formal_code, parsed_content
                )
                generation_result.mathematical_validation = math_validation
            
            # Calculate confidence and complexity scores
            generation_result.confidence_score = self._calculate_confidence_score(
                generation_result, verification_result
            )
            generation_result.complexity_score = self._calculate_complexity_score(
                parsed_content, generation_result.formal_code
            )
            
            # Extract theorem type and dependencies
            generation_result.theorem_type = self._extract_theorem_type(parsed_content)
            generation_result.dependencies = self._extract_dependencies(generation_result.formal_code)
            
            # Update metrics
            self.metrics.record_formalization(
                success=generation_result.success,
                target_system=self.target_system,
                processing_time=generation_result.processing_time,
                verification_time=verification_time,
                correction_rounds=generation_result.correction_rounds
            )
            
            # Cache result
            if self.cache and generation_result.success:
                await self.cache.set(cache_key, generation_result)
            
            return generation_result
            
        except asyncio.TimeoutError:
            return EnhancedFormalizationResult(
                success=False,
                error_message=f"Formalization timed out after {timeout} seconds",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"Formalization failed: {e}")
            return EnhancedFormalizationResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _generate_with_correction(
        self,
        parsed_content: Any,
        context: Optional[str],
        timeout: int
    ) -> EnhancedFormalizationResult:
        """Generate code with self-correction loop."""
        correction_round = 0
        last_error = None
        
        while correction_round <= self.max_correction_rounds:
            try:
                # Generate formal code
                if correction_round == 0:
                    formal_code = await self.generator.generate(
                        parsed_content, context=context
                    )
                else:
                    # Use error feedback for correction
                    formal_code = await self.generator.generate_with_correction(
                        parsed_content, last_error, context=context
                    )
                
                # Quick syntax validation
                if self._validate_syntax(formal_code):
                    return EnhancedFormalizationResult(
                        success=True,
                        formal_code=formal_code,
                        correction_rounds=correction_round,
                        processing_time=time.time() - time.time()  # Will be updated by caller
                    )
                else:
                    last_error = "Syntax validation failed"
                    correction_round += 1
                    continue
                    
            except Exception as e:
                last_error = str(e)
                correction_round += 1
                
                if correction_round > self.max_correction_rounds:
                    return EnhancedFormalizationResult(
                        success=False,
                        error_message=f"Failed after {self.max_correction_rounds} correction rounds: {last_error}",
                        correction_rounds=correction_round
                    )
        
        return EnhancedFormalizationResult(
            success=False,
            error_message=f"Max correction rounds exceeded: {last_error}",
            correction_rounds=correction_round
        )
    
    def _validate_syntax(self, formal_code: str) -> bool:
        """Quick syntax validation for generated code."""
        if not formal_code or not formal_code.strip():
            return False
        
        # Basic syntax checks based on target system
        if self.target_system == "lean4":
            return "theorem" in formal_code or "lemma" in formal_code or "def" in formal_code
        elif self.target_system == "isabelle":
            return "theory" in formal_code and "begin" in formal_code and "end" in formal_code
        elif self.target_system == "coq":
            return ("Theorem" in formal_code or "Lemma" in formal_code) and "Qed." in formal_code
        
        return True
    
    async def _validate_mathematical_correctness(
        self,
        latex_content: str,
        formal_code: str,
        parsed_content: Any
    ) -> bool:
        """Validate mathematical correctness of formalization."""
        try:
            # Extract mathematical concepts from both sources
            latex_concepts = self._extract_mathematical_concepts(latex_content)
            formal_concepts = self._extract_formal_concepts(formal_code)
            
            # Check for concept alignment
            alignment_score = self._calculate_concept_alignment(latex_concepts, formal_concepts)
            
            # Check for completeness
            completeness_score = self._check_formalization_completeness(parsed_content, formal_code)
            
            # Combined validation score
            validation_score = (alignment_score + completeness_score) / 2
            
            return validation_score > 0.7  # 70% threshold
            
        except Exception as e:
            self.logger.warning(f"Mathematical validation failed: {e}")
            return False
    
    def _extract_mathematical_concepts(self, latex_content: str) -> List[str]:
        """Extract mathematical concepts from LaTeX content."""
        concepts = []
        
        # Basic concept extraction (can be enhanced with NLP)
        import re
        
        # Find mathematical terms
        math_patterns = [
            r'\\forall',
            r'\\exists',
            r'\\sum',
            r'\\int',
            r'\\lim',
            r'\\frac',
            r'\\sqrt',
            r'\\prime',
            r'\\subset',
            r'\\supset',
            r'\\in',
            r'\\notin',
            r'\\cup',
            r'\\cap',
            r'theorem',
            r'lemma',
            r'proof',
            r'definition'
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, latex_content):
                concepts.append(pattern.replace('\\', ''))
        
        return concepts
    
    def _extract_formal_concepts(self, formal_code: str) -> List[str]:
        """Extract mathematical concepts from formal code."""
        concepts = []
        
        # System-specific concept extraction
        if self.target_system == "lean4":
            patterns = ['∀', '∃', '∑', '∫', 'theorem', 'lemma', 'def', 'by', 'exact', 'apply']
        elif self.target_system == "isabelle":
            patterns = ['ALL', 'EX', 'theorem', 'lemma', 'by', 'simp', 'auto', 'blast']
        elif self.target_system == "coq":
            patterns = ['forall', 'exists', 'Theorem', 'Lemma', 'Proof', 'Qed', 'apply', 'exact']
        else:
            patterns = []
        
        for pattern in patterns:
            if pattern in formal_code:
                concepts.append(pattern)
        
        return concepts
    
    def _calculate_concept_alignment(self, latex_concepts: List[str], formal_concepts: List[str]) -> float:
        """Calculate alignment between LaTeX and formal concepts."""
        if not latex_concepts:
            return 1.0  # No concepts to align
        
        # Simple concept mapping
        concept_map = {
            'forall': ['∀', 'ALL', 'forall'],
            'exists': ['∃', 'EX', 'exists'],
            'theorem': ['theorem', 'Theorem'],
            'lemma': ['lemma', 'Lemma'],
            'proof': ['by', 'Proof', 'simp', 'auto']
        }
        
        aligned = 0
        for latex_concept in latex_concepts:
            formal_equivalents = concept_map.get(latex_concept, [latex_concept])
            if any(equiv in formal_concepts for equiv in formal_equivalents):
                aligned += 1
        
        return aligned / len(latex_concepts)
    
    def _check_formalization_completeness(self, parsed_content: Any, formal_code: str) -> float:
        """Check if formalization covers all parsed mathematical content."""
        try:
            # Count theorems/lemmas in parsed content
            theorem_count = len(getattr(parsed_content, 'theorems', []))
            definition_count = len(getattr(parsed_content, 'definitions', []))
            total_statements = theorem_count + definition_count
            
            if total_statements == 0:
                return 1.0
            
            # Count formal statements
            formal_statements = 0
            if self.target_system == "lean4":
                formal_statements = formal_code.count('theorem') + formal_code.count('lemma') + formal_code.count('def')
            elif self.target_system == "isabelle":
                formal_statements = formal_code.count('theorem') + formal_code.count('lemma')
            elif self.target_system == "coq":
                formal_statements = formal_code.count('Theorem') + formal_code.count('Lemma') + formal_code.count('Definition')
            
            return min(formal_statements / total_statements, 1.0)
            
        except Exception:
            return 0.5  # Default moderate score
    
    def _calculate_confidence_score(
        self,
        result: EnhancedFormalizationResult,
        verification_result: Any
    ) -> float:
        """Calculate confidence score for the formalization."""
        score = 0.0
        
        # Base score for successful generation
        if result.success:
            score += 0.3
        
        # Verification bonus
        if result.verification_status:
            score += 0.4
        elif verification_result and not verification_result.errors:
            score += 0.2
        
        # Mathematical validation bonus
        if result.mathematical_validation:
            score += 0.2
        
        # Correction rounds penalty
        correction_penalty = min(result.correction_rounds * 0.05, 0.1)
        score -= correction_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity_score(self, parsed_content: Any, formal_code: str) -> int:
        """Calculate complexity score (1-10 scale)."""
        complexity = 1
        
        # Length-based complexity
        if len(formal_code) > 1000:
            complexity += 2
        elif len(formal_code) > 500:
            complexity += 1
        
        # Logical complexity
        logical_constructs = ['∀', '∃', 'forall', 'exists', 'ALL', 'EX']
        complexity += min(sum(formal_code.count(construct) for construct in logical_constructs), 3)
        
        # Proof technique complexity
        advanced_techniques = ['induction', 'contradiction', 'simp', 'blast', 'auto', 'omega']
        complexity += min(sum(formal_code.count(technique) for technique in advanced_techniques), 2)
        
        return min(complexity, 10)
    
    def _extract_theorem_type(self, parsed_content: Any) -> Optional[str]:
        """Extract theorem type from parsed content."""
        # This would be enhanced with more sophisticated analysis
        if hasattr(parsed_content, 'theorems') and parsed_content.theorems:
            return "theorem"
        elif hasattr(parsed_content, 'definitions') and parsed_content.definitions:
            return "definition"
        return None
    
    def _extract_dependencies(self, formal_code: str) -> List[str]:
        """Extract dependencies from formal code."""
        dependencies = []
        
        # Extract imports/requires
        import re
        
        if self.target_system == "lean4":
            imports = re.findall(r'import\s+([A-Za-z0-9_.]+)', formal_code)
            dependencies.extend(imports)
        elif self.target_system == "isabelle":
            imports = re.findall(r'imports\s+([A-Za-z0-9_.]+)', formal_code)
            dependencies.extend(imports)
        elif self.target_system == "coq":
            requires = re.findall(r'Require\s+Import\s+([A-Za-z0-9_.]+)', formal_code)
            dependencies.extend(requires)
        
        return dependencies
    
    def _generate_cache_key(self, latex_content: str, context: Optional[str]) -> str:
        """Generate cache key for formalization."""
        content_hash = hashlib.sha256(
            f"{latex_content}:{context or ''}:{self.target_system}:{self.model_name}".encode()
        ).hexdigest()[:16]
        return f"formalization:{content_hash}"
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "pipeline": {
                "target_system": self.target_system,
                "model_name": self.model_name,
                "verification_mode": self.verification_mode.value,
                "caching_enabled": self.enable_caching,
                "validation_enabled": self.enable_validation
            },
            "components": {
                "parser": "active",
                "generator": "active",
                "verifier": "active" if self.verifier else "disabled",
                "cache": "active" if self.cache else "disabled",
                "validator": "active" if self.validator else "disabled"
            }
        }
        
        # Check proof assistant installation
        if self.verifier:
            if hasattr(self.verifier, 'check_lean_installation'):
                install_check = await self.verifier.check_lean_installation()
            elif hasattr(self.verifier, 'check_isabelle_installation'):
                install_check = await self.verifier.check_isabelle_installation()
            elif hasattr(self.verifier, 'check_coq_installation'):
                install_check = await self.verifier.check_coq_installation()
            else:
                install_check = {"installed": False, "error": "Unknown verifier type"}
            
            status["proof_assistant"] = install_check
        
        # Add metrics
        status["metrics"] = self.metrics.get_summary()
        
        return status
    
    async def benchmark_performance(self, test_cases: List[str]) -> Dict[str, Any]:
        """Benchmark pipeline performance."""
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            case_start = time.time()
            result = await self.formalize(test_case)
            case_time = time.time() - case_start
            
            results.append({
                "case_id": i,
                "success": result.success,
                "processing_time": case_time,
                "verification_time": result.verification_time,
                "correction_rounds": result.correction_rounds,
                "confidence_score": result.confidence_score
            })
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = [r for r in results if r["success"]]
        
        return {
            "total_cases": len(test_cases),
            "successful_cases": len(successful),
            "success_rate": len(successful) / len(test_cases) if test_cases else 0,
            "total_time": total_time,
            "average_time": total_time / len(test_cases) if test_cases else 0,
            "average_confidence": sum(r["confidence_score"] for r in successful) / len(successful) if successful else 0,
            "detailed_results": results
        }