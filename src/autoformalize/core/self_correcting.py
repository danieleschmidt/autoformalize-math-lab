"""Self-correcting formalization pipeline.

This module provides a pipeline that iteratively refines formal code
based on verification feedback and error correction.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .pipeline import FormalizationPipeline, FormalizationResult
from .config import FormalizationConfig
from ..core.exceptions import FormalizationError
from ..utils.logging_config import setup_logger


@dataclass
class CorrectionAttempt:
    """Represents a single correction attempt."""
    round_number: int
    error_messages: List[str]
    correction_prompt: str
    generated_code: Optional[str] = None
    verification_success: bool = False
    processing_time: float = 0.0


class SelfCorrectingPipeline(FormalizationPipeline):
    """Pipeline with self-correction capabilities.
    
    This pipeline extends the basic FormalizationPipeline with the ability
    to iteratively improve generated code based on verification feedback.
    """
    
    def __init__(
        self,
        target_system: str = "lean4",
        model: str = "gpt-4", 
        config: Optional[FormalizationConfig] = None,
        max_correction_rounds: int = 5,
        correction_prompts: Optional[List[str]] = None,
        verifier_timeout: int = 30
    ):
        """Initialize self-correcting pipeline.
        
        Args:
            target_system: Target proof assistant system
            model: LLM model to use
            config: Configuration object
            max_correction_rounds: Maximum number of correction attempts
            correction_prompts: List of correction prompt templates
            verifier_timeout: Timeout for verification in seconds
        """
        super().__init__(target_system, model, config)
        self.max_correction_rounds = max_correction_rounds
        self.verifier_timeout = verifier_timeout
        self.correction_attempts: List[CorrectionAttempt] = []
        
        # Default correction prompts
        self.correction_prompts = correction_prompts or [
            "Fix syntax errors in the following code",
            "Resolve type mismatches in the following code", 
            "Find and add missing imports for the following code",
            "Simplify the proof tactics in the following code",
            "Correct logical errors in the following proof"
        ]
    
    async def formalize_with_feedback(
        self,
        latex_content: str,
        verify: bool = True,
        verbose: bool = False
    ) -> FormalizationResult:
        """Formalize with iterative correction based on feedback.
        
        Args:
            latex_content: LaTeX content to formalize
            verify: Whether to verify generated code
            verbose: Whether to show correction attempts
            
        Returns:
            FormalizationResult with correction history
        """
        self.correction_attempts.clear()
        
        # Initial formalization attempt
        result = await self.formalize(latex_content, verify=verify)
        
        if not verify or result.verification_status or not result.success:
            return result
        
        # If verification failed, attempt corrections
        if verbose:
            print(f"Initial verification failed. Starting correction process...")
        
        current_code = result.formal_code
        correction_round = 0
        
        while (correction_round < self.max_correction_rounds and 
               current_code and 
               not result.verification_status):
            
            correction_round += 1
            
            if verbose:
                print(f"Correction round {correction_round}/{self.max_correction_rounds}")
            
            # Get verification errors
            if self.verifier:
                verification_result = await self.verifier.verify_detailed(
                    current_code, 
                    timeout=self.verifier_timeout
                )
                error_messages = verification_result.errors
            else:
                error_messages = ["Verification failed (no verifier available)"]
            
            # Create correction attempt
            attempt = CorrectionAttempt(
                round_number=correction_round,
                error_messages=error_messages
            )
            
            # Generate correction prompt
            prompt_template = self.correction_prompts[
                min(correction_round - 1, len(self.correction_prompts) - 1)
            ]
            
            attempt.correction_prompt = self._create_correction_prompt(
                prompt_template,
                current_code,
                error_messages
            )
            
            # Generate corrected code
            start_time = asyncio.get_event_loop().time()
            
            try:
                corrected_code = await self._generate_correction(attempt.correction_prompt)
                attempt.generated_code = corrected_code
                
                if corrected_code:
                    current_code = corrected_code
                    
                    # Verify corrected code
                    if self.verifier:
                        verification_result = await self.verifier.verify_detailed(
                            current_code,
                            timeout=self.verifier_timeout
                        )
                        attempt.verification_success = verification_result.success
                        
                        if verification_result.success:
                            result.formal_code = current_code
                            result.verification_status = True
                            result.correction_rounds = correction_round
                            break
                
            except Exception as e:
                self.logger.warning(f"Correction round {correction_round} failed: {e}")
            
            attempt.processing_time = asyncio.get_event_loop().time() - start_time
            self.correction_attempts.append(attempt)
            
            if verbose:
                status = "✅" if attempt.verification_success else "❌"
                print(f"  Round {correction_round}: {status}")
        
        # Update final result
        result.correction_rounds = correction_round
        result.metadata = result.metadata or {}
        result.metadata['correction_attempts'] = len(self.correction_attempts)
        result.metadata['correction_history'] = [
            {
                'round': attempt.round_number,
                'success': attempt.verification_success,
                'processing_time': attempt.processing_time,
                'error_count': len(attempt.error_messages)
            }
            for attempt in self.correction_attempts
        ]
        
        return result
    
    def _create_correction_prompt(
        self,
        template: str,
        code: str,
        errors: List[str]
    ) -> str:
        """Create a correction prompt from template and error information.
        
        Args:
            template: Correction prompt template
            code: Current code that needs correction
            errors: List of error messages
            
        Returns:
            Formatted correction prompt
        """
        error_summary = "\n".join(f"- {error}" for error in errors[:5])  # Limit errors
        
        return f"""{template}:

Current code:
```{self.target_system.value}
{code}
```

Errors to fix:
{error_summary}

Please provide corrected code that addresses these errors.
Generate only valid {self.target_system.value} code.
"""
    
    async def _generate_correction(self, prompt: str) -> Optional[str]:
        """Generate corrected code using the LLM.
        
        Args:
            prompt: Correction prompt
            
        Returns:
            Corrected code or None if generation failed
        """
        try:
            if hasattr(self.generator, '_call_llm'):
                response = await self.generator._call_llm(prompt)
                return self.generator._extract_lean_code(response)
            else:
                self.logger.warning("Generator does not support correction")
                return None
                
        except Exception as e:
            self.logger.error(f"Correction generation failed: {e}")
            return None
    
    def get_correction_summary(self) -> Dict[str, Any]:
        """Get summary of correction attempts.
        
        Returns:
            Dictionary with correction statistics
        """
        if not self.correction_attempts:
            return {"correction_attempts": 0}
        
        successful_attempts = sum(1 for attempt in self.correction_attempts 
                                if attempt.verification_success)
        
        total_time = sum(attempt.processing_time for attempt in self.correction_attempts)
        
        return {
            "correction_attempts": len(self.correction_attempts),
            "successful_corrections": successful_attempts,
            "total_correction_time": total_time,
            "average_correction_time": total_time / len(self.correction_attempts),
            "success_rate": successful_attempts / len(self.correction_attempts)
        }