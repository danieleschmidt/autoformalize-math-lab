"""Self-correction and error recovery system.

This module provides automatic error correction and recovery mechanisms
for the formalization pipeline, including LLM-based error fixing.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .exceptions import GenerationError, VerificationError
from ..utils.logging_config import setup_logger


class ErrorType(Enum):
    """Types of errors that can be corrected."""
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    UNDEFINED_REFERENCE = "undefined_reference"
    IMPORT_ERROR = "import_error"
    PROOF_INCOMPLETE = "proof_incomplete"
    UNKNOWN = "unknown"


@dataclass
class ErrorPattern:
    """Pattern for recognizing and classifying errors."""
    error_type: ErrorType
    pattern: str
    description: str
    fix_template: Optional[str] = None
    confidence: float = 1.0


@dataclass
class CorrectionAttempt:
    """Represents a single correction attempt."""
    attempt_number: int
    error_type: ErrorType
    original_error: str
    correction_strategy: str
    corrected_code: str
    success: bool = False
    new_errors: List[str] = None
    
    def __post_init__(self):
        if self.new_errors is None:
            self.new_errors = []


class ErrorAnalyzer:
    """Analyzes errors and suggests corrections."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self._setup_error_patterns()
    
    def _setup_error_patterns(self) -> None:
        """Setup error recognition patterns for different proof assistants."""
        
        # Lean 4 error patterns
        self.lean_patterns = [
            ErrorPattern(
                ErrorType.SYNTAX_ERROR,
                r"expected.*got.*",
                "Lean syntax error - unexpected token",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.TYPE_ERROR,
                r"type mismatch.*expected.*got.*",
                "Lean type mismatch error",
                confidence=0.95
            ),
            ErrorPattern(
                ErrorType.UNDEFINED_REFERENCE,
                r"unknown identifier.*",
                "Lean undefined identifier",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.IMPORT_ERROR,
                r"file.*not found",
                "Lean import/file not found error",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.PROOF_INCOMPLETE,
                r"unsolved goals.*",
                "Lean incomplete proof",
                confidence=0.8
            )
        ]
        
        # Isabelle error patterns
        self.isabelle_patterns = [
            ErrorPattern(
                ErrorType.SYNTAX_ERROR,
                r"Inner syntax error.*",
                "Isabelle syntax error",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.TYPE_ERROR,
                r"Type unification failed.*",
                "Isabelle type unification error",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.UNDEFINED_REFERENCE,
                r"Unknown fact.*",
                "Isabelle undefined fact/theorem",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.PROOF_INCOMPLETE,
                r"Failed to finish proof.*",
                "Isabelle incomplete proof",
                confidence=0.8
            )
        ]
        
        # Coq error patterns
        self.coq_patterns = [
            ErrorPattern(
                ErrorType.SYNTAX_ERROR,
                r"Syntax error.*",
                "Coq syntax error",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.TYPE_ERROR,
                r"The term .* has type .* while it is expected to have type.*",
                "Coq type mismatch error",
                confidence=0.95
            ),
            ErrorPattern(
                ErrorType.UNDEFINED_REFERENCE,
                r"The reference .* was not found.*",
                "Coq undefined reference",
                confidence=0.9
            ),
            ErrorPattern(
                ErrorType.PROOF_INCOMPLETE,
                r"This subproof is complete, but there are still unfocused goals.*",
                "Coq incomplete proof - unfocused goals",
                confidence=0.8
            )
        ]
    
    def analyze_errors(self, errors: List[str], target_system: str) -> List[Tuple[ErrorType, str, float]]:
        """Analyze errors and classify them.
        
        Args:
            errors: List of error messages
            target_system: Target proof assistant system
            
        Returns:
            List of tuples (error_type, error_message, confidence)
        """
        classified_errors = []
        
        # Select appropriate patterns
        patterns = []
        if target_system == "lean4":
            patterns = self.lean_patterns
        elif target_system == "isabelle":
            patterns = self.isabelle_patterns
        elif target_system == "coq":
            patterns = self.coq_patterns
        
        for error in errors:
            error_type = ErrorType.UNKNOWN
            max_confidence = 0.0
            
            # Try to match error patterns
            for pattern in patterns:
                if re.search(pattern.pattern, error, re.IGNORECASE):
                    if pattern.confidence > max_confidence:
                        error_type = pattern.error_type
                        max_confidence = pattern.confidence
            
            classified_errors.append((error_type, error, max_confidence))
            
        return classified_errors
    
    def suggest_correction_strategies(
        self,
        error_type: ErrorType,
        error_message: str,
        target_system: str
    ) -> List[str]:
        """Suggest correction strategies for a specific error.
        
        Args:
            error_type: Type of error
            error_message: The error message
            target_system: Target proof assistant system
            
        Returns:
            List of correction strategy descriptions
        """
        strategies = []
        
        if error_type == ErrorType.SYNTAX_ERROR:
            strategies.extend([
                "Fix syntax by correcting punctuation and keywords",
                "Add missing parentheses or brackets",
                "Correct variable and function names",
                "Fix indentation and formatting"
            ])
        
        elif error_type == ErrorType.TYPE_ERROR:
            strategies.extend([
                "Add missing type annotations",
                "Cast types explicitly",
                "Fix type mismatches in function applications",
                "Correct generic type parameters"
            ])
        
        elif error_type == ErrorType.UNDEFINED_REFERENCE:
            strategies.extend([
                "Add missing imports",
                "Define missing functions or theorems",
                "Fix typos in identifiers",
                "Add required namespaces"
            ])
        
        elif error_type == ErrorType.IMPORT_ERROR:
            strategies.extend([
                "Add missing import statements",
                "Fix import paths",
                "Check module availability",
                "Update dependency versions"
            ])
        
        elif error_type == ErrorType.PROOF_INCOMPLETE:
            strategies.extend([
                "Complete proof with additional tactics",
                "Add missing cases in proof",
                "Strengthen hypotheses",
                "Use different proof strategy"
            ])
        
        # Add system-specific strategies
        if target_system == "lean4":
            strategies.append("Use Lean 4 specific tactics (simp, rw, exact, apply)")
        elif target_system == "isabelle":
            strategies.append("Use Isabelle methods (auto, simp, blast, force)")
        elif target_system == "coq":
            strategies.append("Use Coq tactics (auto, trivial, lia, omega)")
        
        return strategies


class SelfCorrectingGenerator:
    """Generator with self-correction capabilities."""
    
    def __init__(self, base_generator, verifier, max_attempts: int = 3):
        """Initialize self-correcting generator.
        
        Args:
            base_generator: Base generator (Lean4Generator, etc.)
            verifier: Corresponding verifier
            max_attempts: Maximum correction attempts
        """
        self.base_generator = base_generator
        self.verifier = verifier
        self.max_attempts = max_attempts
        self.error_analyzer = ErrorAnalyzer()
        self.logger = setup_logger(__name__)
        
        # Correction history
        self.correction_history: List[CorrectionAttempt] = []
    
    async def generate_with_correction(self, parsed_content) -> Tuple[str, List[CorrectionAttempt]]:
        """Generate code with automatic error correction.
        
        Args:
            parsed_content: Parsed mathematical content
            
        Returns:
            Tuple of (final_code, correction_attempts)
        """
        self.correction_history.clear()
        
        try:
            # Initial generation
            self.logger.info("Generating initial code...")
            current_code = await self.base_generator.generate(parsed_content)
            
            # Verify initial code
            verification_result = await self.verifier.verify_detailed(current_code)
            
            if verification_result.success:
                self.logger.info("Initial code verification successful")
                return current_code, self.correction_history
            
            # If verification failed, attempt corrections
            self.logger.warning(f"Initial verification failed with {len(verification_result.errors)} errors")
            
            for attempt_num in range(1, self.max_attempts + 1):
                self.logger.info(f"Starting correction attempt {attempt_num}/{self.max_attempts}")
                
                # Analyze errors
                classified_errors = self.error_analyzer.analyze_errors(
                    verification_result.errors,
                    self._get_target_system()
                )
                
                if not classified_errors:
                    self.logger.warning("No errors to classify, stopping correction attempts")
                    break
                
                # Attempt correction
                corrected_code = await self._attempt_correction(
                    current_code,
                    classified_errors,
                    attempt_num
                )
                
                if corrected_code is None:
                    self.logger.error(f"Correction attempt {attempt_num} failed to generate code")
                    break
                
                # Verify corrected code
                new_verification = await self.verifier.verify_detailed(corrected_code)
                
                # Record attempt
                primary_error_type = classified_errors[0][0] if classified_errors else ErrorType.UNKNOWN
                attempt = CorrectionAttempt(
                    attempt_number=attempt_num,
                    error_type=primary_error_type,
                    original_error=verification_result.errors[0] if verification_result.errors else "",
                    correction_strategy=f"LLM-based correction for {primary_error_type.value}",
                    corrected_code=corrected_code,
                    success=new_verification.success,
                    new_errors=new_verification.errors
                )
                self.correction_history.append(attempt)
                
                if new_verification.success:
                    self.logger.info(f"Correction successful on attempt {attempt_num}")
                    return corrected_code, self.correction_history
                
                # Update for next iteration
                current_code = corrected_code
                verification_result = new_verification
                
                self.logger.warning(f"Attempt {attempt_num} still has {len(verification_result.errors)} errors")
            
            # All attempts exhausted
            self.logger.error(f"All {self.max_attempts} correction attempts failed")
            return current_code, self.correction_history
            
        except Exception as e:
            self.logger.error(f"Error during correction process: {e}")
            raise GenerationError(f"Self-correction failed: {e}")
    
    async def _attempt_correction(
        self,
        faulty_code: str,
        classified_errors: List[Tuple[ErrorType, str, float]],
        attempt_num: int
    ) -> Optional[str]:
        """Attempt to correct code based on classified errors.
        
        Args:
            faulty_code: Code with errors
            classified_errors: Classified error information
            attempt_num: Current attempt number
            
        Returns:
            Corrected code or None if correction failed
        """
        try:
            # Create correction prompt
            correction_prompt = self._create_correction_prompt(
                faulty_code,
                classified_errors,
                attempt_num
            )
            
            # Use base generator's LLM to generate correction
            if hasattr(self.base_generator, '_call_llm'):
                corrected_code_response = await self.base_generator._call_llm(correction_prompt)
                
                # Extract corrected code
                corrected_code = self._extract_corrected_code(corrected_code_response)
                
                return corrected_code
            else:
                self.logger.error("Base generator doesn't support LLM calls for correction")
                return None
                
        except Exception as e:
            self.logger.error(f"Correction attempt failed: {e}")
            return None
    
    def _create_correction_prompt(
        self,
        faulty_code: str,
        classified_errors: List[Tuple[ErrorType, str, float]],
        attempt_num: int
    ) -> str:
        """Create a prompt for error correction.
        
        Args:
            faulty_code: Code with errors
            classified_errors: Classified error information
            attempt_num: Current attempt number
            
        Returns:
            Correction prompt
        """
        error_descriptions = []
        for error_type, error_msg, confidence in classified_errors:
            error_descriptions.append(f"- {error_type.value}: {error_msg} (confidence: {confidence:.2f})")
        
        errors_text = "\n".join(error_descriptions)
        
        system_name = self._get_target_system().upper()
        
        prompt = f"""Fix the following {system_name} code that has verification errors.

ORIGINAL CODE:
```{system_name.lower()}
{faulty_code}
```

ERRORS FOUND:
{errors_text}

CORRECTION INSTRUCTIONS:
1. Analyze each error carefully
2. Fix syntax errors, type mismatches, and undefined references
3. Ensure all imports are correct and complete
4. Complete any incomplete proofs
5. Maintain the mathematical meaning and structure
6. Generate only valid {system_name} code

This is correction attempt #{attempt_num}. Please provide the complete corrected code."""

        return prompt
    
    def _extract_corrected_code(self, response: str) -> str:
        """Extract corrected code from LLM response.
        
        Args:
            response: LLM response containing corrected code
            
        Returns:
            Extracted corrected code
        """
        # Use base generator's extraction method if available
        if hasattr(self.base_generator, '_extract_lean_code'):
            return self.base_generator._extract_lean_code(response)
        elif hasattr(self.base_generator, '_extract_isabelle_code'):
            return self.base_generator._extract_isabelle_code(response)
        elif hasattr(self.base_generator, '_extract_coq_code'):
            return self.base_generator._extract_coq_code(response)
        else:
            # Fallback extraction
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
            if code_blocks:
                return code_blocks[0].strip()
            return response.strip()
    
    def _get_target_system(self) -> str:
        """Get target system name from generator type."""
        generator_name = type(self.base_generator).__name__.lower()
        if 'lean' in generator_name:
            return 'lean4'
        elif 'isabelle' in generator_name:
            return 'isabelle'
        elif 'coq' in generator_name:
            return 'coq'
        else:
            return 'unknown'
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get statistics about correction attempts.
        
        Returns:
            Dictionary with correction statistics
        """
        if not self.correction_history:
            return {
                "total_attempts": 0,
                "successful_attempts": 0,
                "success_rate": 0.0,
                "error_types_encountered": [],
                "most_common_error_type": None
            }
        
        total_attempts = len(self.correction_history)
        successful_attempts = sum(1 for attempt in self.correction_history if attempt.success)
        
        error_types = [attempt.error_type for attempt in self.correction_history]
        error_type_counts = {}
        for error_type in error_types:
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        most_common_error = max(error_type_counts.items(), key=lambda x: x[1]) if error_type_counts else (None, 0)
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0.0,
            "error_types_encountered": list(error_type_counts.keys()),
            "error_type_distribution": error_type_counts,
            "most_common_error_type": most_common_error[0].value if most_common_error[0] else None,
            "correction_history": [
                {
                    "attempt": attempt.attempt_number,
                    "error_type": attempt.error_type.value,
                    "success": attempt.success,
                    "strategy": attempt.correction_strategy
                }
                for attempt in self.correction_history
            ]
        }