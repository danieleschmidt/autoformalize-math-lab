"""Enhanced mathematical validation utilities.

This module provides comprehensive validation for mathematical content,
formal proofs, and system configurations to ensure correctness and reliability.
"""

import re
import ast
import sympy
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .logging_config import setup_logger


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    RESEARCH = "research"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    valid: bool
    score: float = 0.0
    issues: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}


class MathematicalValidator:
    """Comprehensive mathematical content validator."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize validator.
        
        Args:
            validation_level: Strictness level for validation
        """
        self.validation_level = validation_level
        self.logger = setup_logger(__name__)
        
        # Load mathematical knowledge base
        self.mathematical_constants = self._load_mathematical_constants()
        self.common_functions = self._load_common_functions()
        self.proof_techniques = self._load_proof_techniques()
    
    def _load_mathematical_constants(self) -> Set[str]:
        """Load known mathematical constants."""
        return {
            'π', 'pi', 'e', 'euler', 'γ', 'gamma', 'φ', 'phi', 'golden_ratio',
            '∞', 'infinity', 'inf', '√2', 'sqrt2', '√3', 'sqrt3', '√5', 'sqrt5',
            'ℕ', 'naturals', 'ℤ', 'integers', 'ℚ', 'rationals', 'ℝ', 'reals',
            'ℂ', 'complex', 'ℙ', 'primes'
        }
    
    def _load_common_functions(self) -> Set[str]:
        """Load common mathematical functions."""
        return {
            'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
            'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
            'exp', 'log', 'ln', 'lg', 'sqrt', 'cbrt',
            'floor', 'ceil', 'round', 'abs', 'sign', 'max', 'min',
            'gcd', 'lcm', 'mod', 'factorial', 'choose', 'binomial',
            'derivative', 'integral', 'sum', 'product', 'limit'
        }
    
    def _load_proof_techniques(self) -> Set[str]:
        """Load common proof techniques."""
        return {
            'induction', 'contradiction', 'contrapositive', 'direct',
            'existence', 'uniqueness', 'construction', 'pigeonhole',
            'diagonalization', 'reduction', 'exhaustion', 'probabilistic'
        }
    
    async def validate_latex_content(
        self,
        latex_content: str,
        context: Optional[str] = None
    ) -> ValidationResult:
        """Validate LaTeX mathematical content.
        
        Args:
            latex_content: LaTeX content to validate
            context: Additional context for validation
            
        Returns:
            Validation result with detailed feedback
        """
        issues = []
        warnings = []
        suggestions = []
        score = 1.0
        
        try:
            # Basic syntax validation
            syntax_result = self._validate_latex_syntax(latex_content)
            if not syntax_result.valid:
                issues.extend(syntax_result.issues)
                score *= 0.5
            else:
                warnings.extend(syntax_result.warnings)
                suggestions.extend(syntax_result.suggestions)
            
            # Mathematical content validation
            math_result = self._validate_mathematical_content(latex_content)
            if not math_result.valid:
                issues.extend(math_result.issues)
                score *= 0.7
            else:
                warnings.extend(math_result.warnings)
                suggestions.extend(math_result.suggestions)
            
            # Logical structure validation
            logic_result = self._validate_logical_structure(latex_content)
            if not logic_result.valid:
                issues.extend(logic_result.issues)
                score *= 0.8
            else:
                warnings.extend(logic_result.warnings)
                suggestions.extend(logic_result.suggestions)
            
            # Domain-specific validation
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.RESEARCH]:
                domain_result = self._validate_domain_specific(latex_content)
                warnings.extend(domain_result.warnings)
                suggestions.extend(domain_result.suggestions)
            
            return ValidationResult(
                valid=len(issues) == 0,
                score=score,
                issues=issues,
                warnings=warnings,
                suggestions=suggestions,
                metadata={
                    "validation_level": self.validation_level.value,
                    "content_length": len(latex_content),
                    "mathematical_elements": self._count_mathematical_elements(latex_content)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                valid=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"]
            )
    
    def _validate_latex_syntax(self, latex_content: str) -> ValidationResult:
        """Validate LaTeX syntax."""
        issues = []
        warnings = []
        suggestions = []
        
        # Check for balanced braces
        brace_count = latex_content.count('{') - latex_content.count('}')
        if brace_count != 0:
            issues.append(f"Unbalanced braces: {abs(brace_count)} {'extra opening' if brace_count > 0 else 'missing opening'} braces")
        
        # Check for balanced brackets
        bracket_count = latex_content.count('[') - latex_content.count(']')
        if bracket_count != 0:
            warnings.append(f"Unbalanced brackets: {abs(bracket_count)} {'extra opening' if bracket_count > 0 else 'missing opening'} brackets")
        
        # Check for balanced parentheses
        paren_count = latex_content.count('(') - latex_content.count(')')
        if paren_count != 0:
            issues.append(f"Unbalanced parentheses: {abs(paren_count)} {'extra opening' if paren_count > 0 else 'missing opening'} parentheses")
        
        # Check for common LaTeX errors
        if '\\begin{' in latex_content:
            begins = re.findall(r'\\begin\{([^}]+)\}', latex_content)
            ends = re.findall(r'\\end\{([^}]+)\}', latex_content)
            
            for env in begins:
                if env not in ends:
                    issues.append(f"Missing \\end{{{env}}} for \\begin{{{env}}}")
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_mathematical_content(self, latex_content: str) -> ValidationResult:
        """Validate mathematical content for consistency."""
        issues = []
        warnings = []
        suggestions = []
        
        # Check for mathematical environments
        math_envs = ['equation', 'align', 'gather', 'multline', 'theorem', 'lemma', 'proof']
        has_math = any(env in latex_content for env in math_envs) or '$' in latex_content
        
        if not has_math:
            warnings.append("No mathematical content detected")
        
        # Check for empty fractions and roots
        if '\\frac{}' in latex_content:
            issues.append("Empty fraction detected")
        
        if '\\sqrt{}' in latex_content:
            issues.append("Empty square root detected")
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_logical_structure(self, latex_content: str) -> ValidationResult:
        """Validate logical structure of mathematical content."""
        issues = []
        warnings = []
        suggestions = []
        
        # Check theorem-proof structure
        theorems = len(re.findall(r'\\begin\{theorem\}', latex_content))
        proofs = len(re.findall(r'\\begin\{proof\}', latex_content))
        
        if theorems > proofs:
            warnings.append(f"{theorems - proofs} theorem(s) without proof")
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_domain_specific(self, latex_content: str) -> ValidationResult:
        """Validate domain-specific mathematical content."""
        warnings = []
        suggestions = []
        
        # Basic domain detection and validation
        if 'group' in latex_content.lower() and 'identity' not in latex_content.lower():
            suggestions.append("Group theory: consider mentioning identity element")
        
        return ValidationResult(
            valid=True,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _count_mathematical_elements(self, content: str) -> Dict[str, int]:
        """Count various mathematical elements in content."""
        return {
            'equations': content.count('$'),
            'theorems': content.count('theorem'),
            'lemmas': content.count('lemma'),
            'proofs': content.count('proof'),
            'definitions': content.count('definition')
        }