"""Cross-system translation for formal proofs.

This module provides functionality to translate formal proofs between
different proof assistant systems (Lean 4, Isabelle, Coq, etc.).
"""

import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

from .pipeline import FormalizationPipeline, TargetSystem
from .config import FormalizationConfig
from ..core.exceptions import FormalizationError, UnsupportedSystemError
from ..utils.logging_config import setup_logger


class TranslationDirection(Enum):
    """Supported translation directions."""
    LEAN_TO_ISABELLE = "lean4_to_isabelle"
    LEAN_TO_COQ = "lean4_to_coq"
    LEAN_TO_AGDA = "lean4_to_agda"
    ISABELLE_TO_LEAN = "isabelle_to_lean4"
    COQ_TO_LEAN = "coq_to_lean4"


@dataclass
class TranslationResult:
    """Result of a cross-system translation."""
    success: bool
    source_system: str
    target_system: str
    source_code: str
    translated_code: Optional[str] = None
    verification_results: Dict[str, bool] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.verification_results is None:
            self.verification_results = {}


class CrossSystemTranslator:
    """Translator for formal proofs between proof assistant systems.
    
    This class provides functionality to translate formal proofs from one
    proof assistant system to another using LLM-based translation.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        config: Optional[FormalizationConfig] = None
    ):
        """Initialize the cross-system translator.
        
        Args:
            model: LLM model to use for translation
            config: Configuration object
        """
        self.model = model
        self.config = config or FormalizationConfig()
        self.logger = setup_logger(__name__)
        
        # Initialize pipelines for different systems
        self._pipelines: Dict[str, FormalizationPipeline] = {}
        
        # Translation prompt templates
        self.translation_templates = {
            TranslationDirection.LEAN_TO_ISABELLE: self._lean_to_isabelle_template,
            TranslationDirection.LEAN_TO_COQ: self._lean_to_coq_template,
            TranslationDirection.ISABELLE_TO_LEAN: self._isabelle_to_lean_template,
            TranslationDirection.COQ_TO_LEAN: self._coq_to_lean_template,
        }
    
    def _get_pipeline(self, target_system: str) -> FormalizationPipeline:
        """Get or create a pipeline for the target system.
        
        Args:
            target_system: Target proof assistant system
            
        Returns:
            FormalizationPipeline for the target system
        """
        if target_system not in self._pipelines:
            self._pipelines[target_system] = FormalizationPipeline(
                target_system=target_system,
                model=self.model,
                config=self.config
            )
        return self._pipelines[target_system]
    
    async def translate(
        self,
        source_code: str,
        source_system: str,
        target_system: str,
        verify_result: bool = True
    ) -> TranslationResult:
        """Translate formal code between proof assistant systems.
        
        Args:
            source_code: Source formal proof code
            source_system: Source proof assistant system
            target_system: Target proof assistant system
            verify_result: Whether to verify the translated code
            
        Returns:
            TranslationResult with translation outcome
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"Translating from {source_system} to {target_system}")
            
            # Determine translation direction
            direction = self._get_translation_direction(source_system, target_system)
            if direction not in self.translation_templates:
                raise UnsupportedSystemError(
                    f"Translation from {source_system} to {target_system} not supported"
                )
            
            # Generate translation prompt
            template_func = self.translation_templates[direction]
            prompt = template_func(source_code)
            
            # Get target pipeline
            target_pipeline = self._get_pipeline(target_system)
            
            # Generate translated code
            if hasattr(target_pipeline.generator, '_call_llm'):
                response = await target_pipeline.generator._call_llm(prompt)
                translated_code = target_pipeline.generator._extract_lean_code(response)
            else:
                raise FormalizationError("Target pipeline does not support translation")
            
            if not translated_code:
                raise FormalizationError("Translation generated empty code")
            
            # Verify translated code if requested
            verification_results = {}
            if verify_result and target_pipeline.verifier:
                verification_success = await target_pipeline.verifier.verify(translated_code)
                verification_results[target_system] = verification_success
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return TranslationResult(
                success=True,
                source_system=source_system,
                target_system=target_system,
                source_code=source_code,
                translated_code=translated_code,
                verification_results=verification_results,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Translation failed: {e}")
            
            return TranslationResult(
                success=False,
                source_system=source_system,
                target_system=target_system,
                source_code=source_code,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def lean_to_isabelle(self, lean_code: str) -> str:
        """Translate Lean 4 code to Isabelle/HOL.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            Translated Isabelle/HOL code
        """
        result = await self.translate(lean_code, "lean4", "isabelle")
        if not result.success:
            raise FormalizationError(f"Translation failed: {result.error_message}")
        return result.translated_code
    
    async def lean_to_coq(self, lean_code: str) -> str:
        """Translate Lean 4 code to Coq.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            Translated Coq code
        """
        result = await self.translate(lean_code, "lean4", "coq")
        if not result.success:
            raise FormalizationError(f"Translation failed: {result.error_message}")
        return result.translated_code
    
    async def isabelle_to_lean(self, isabelle_code: str) -> str:
        """Translate Isabelle/HOL code to Lean 4.
        
        Args:
            isabelle_code: Isabelle/HOL source code
            
        Returns:
            Translated Lean 4 code
        """
        result = await self.translate(isabelle_code, "isabelle", "lean4")
        if not result.success:
            raise FormalizationError(f"Translation failed: {result.error_message}")
        return result.translated_code
    
    async def coq_to_lean(self, coq_code: str) -> str:
        """Translate Coq code to Lean 4.
        
        Args:
            coq_code: Coq source code
            
        Returns:
            Translated Lean 4 code
        """
        result = await self.translate(coq_code, "coq", "lean4")
        if not result.success:
            raise FormalizationError(f"Translation failed: {result.error_message}")
        return result.translated_code
    
    async def verify_all(self, code_variants: Dict[str, str]) -> Dict[str, bool]:
        """Verify multiple code variants across different systems.
        
        Args:
            code_variants: Dictionary mapping system names to code
            
        Returns:
            Dictionary mapping system names to verification results
        """
        results = {}
        
        verification_tasks = []
        for system, code in code_variants.items():
            if system in ["lean4", "isabelle", "coq"]:
                pipeline = self._get_pipeline(system)
                if pipeline.verifier:
                    task = pipeline.verifier.verify(code)
                    verification_tasks.append((system, task))
        
        if verification_tasks:
            # Run verifications concurrently
            for system, task in verification_tasks:
                try:
                    result = await task
                    results[system] = result
                except Exception as e:
                    self.logger.warning(f"Verification failed for {system}: {e}")
                    results[system] = False
        
        return results
    
    def _get_translation_direction(self, source: str, target: str) -> TranslationDirection:
        """Determine translation direction from source and target systems.
        
        Args:
            source: Source system name
            target: Target system name
            
        Returns:
            TranslationDirection enum value
        """
        direction_map = {
            ("lean4", "isabelle"): TranslationDirection.LEAN_TO_ISABELLE,
            ("lean4", "coq"): TranslationDirection.LEAN_TO_COQ,
            ("lean4", "agda"): TranslationDirection.LEAN_TO_AGDA,
            ("isabelle", "lean4"): TranslationDirection.ISABELLE_TO_LEAN,
            ("coq", "lean4"): TranslationDirection.COQ_TO_LEAN,
        }
        
        direction = direction_map.get((source, target))
        if not direction:
            raise UnsupportedSystemError(f"Translation from {source} to {target} not supported")
        
        return direction
    
    def _lean_to_isabelle_template(self, lean_code: str) -> str:
        """Create translation prompt from Lean 4 to Isabelle/HOL."""
        return f"""Translate the following Lean 4 code to Isabelle/HOL:

```lean
{lean_code}
```

Please provide:
1. Equivalent Isabelle/HOL code with proper syntax
2. Correct type annotations and theorem statements
3. Appropriate proof methods and tactics
4. Necessary theory imports

Generate only valid Isabelle/HOL code."""
    
    def _lean_to_coq_template(self, lean_code: str) -> str:
        """Create translation prompt from Lean 4 to Coq."""
        return f"""Translate the following Lean 4 code to Coq:

```lean
{lean_code}
```

Please provide:
1. Equivalent Coq code with proper syntax
2. Correct type annotations and theorem statements  
3. Appropriate tactics and proof terms
4. Necessary library imports

Generate only valid Coq code."""
    
    def _isabelle_to_lean_template(self, isabelle_code: str) -> str:
        """Create translation prompt from Isabelle/HOL to Lean 4."""
        return f"""Translate the following Isabelle/HOL code to Lean 4:

```isabelle
{isabelle_code}
```

Please provide:
1. Equivalent Lean 4 code with proper syntax
2. Correct type annotations and theorem statements
3. Appropriate tactics and proof terms
4. Necessary Mathlib imports

Generate only valid Lean 4 code."""
    
    def _coq_to_lean_template(self, coq_code: str) -> str:
        """Create translation prompt from Coq to Lean 4."""
        return f"""Translate the following Coq code to Lean 4:

```coq
{coq_code}
```

Please provide:
1. Equivalent Lean 4 code with proper syntax
2. Correct type annotations and theorem statements
3. Appropriate tactics and proof terms
4. Necessary Mathlib imports

Generate only valid Lean 4 code."""
    
    def get_supported_translations(self) -> List[str]:
        """Get list of supported translation directions.
        
        Returns:
            List of supported translation direction strings
        """
        return [direction.value for direction in self.translation_templates.keys()]
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about translation operations.
        
        Returns:
            Dictionary with translation statistics
        """
        return {
            "supported_directions": len(self.translation_templates),
            "available_systems": list(set(
                direction.value.split('_to_')[0] for direction in self.translation_templates.keys()
            ) | set(
                direction.value.split('_to_')[1] for direction in self.translation_templates.keys()
            )),
            "model": self.model
        }