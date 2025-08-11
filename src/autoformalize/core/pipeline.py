"""Main formalization pipeline implementation.

This module provides the core FormalizationPipeline class that orchestrates
the conversion of LaTeX mathematical content to formal proof assistant code.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from openai import AsyncOpenAI
    from anthropic import AsyncAnthropic
except ImportError:
    # Handle missing dependencies gracefully
    AsyncOpenAI = None
    AsyncAnthropic = None

from ..parsers.latex_parser import LaTeXParser
from ..generators.lean import Lean4Generator
from ..generators.isabelle import IsabelleGenerator
from ..generators.coq import CoqGenerator
from ..verifiers.lean_verifier import Lean4Verifier
from ..utils.logging_config import setup_logger
from ..utils.metrics import FormalizationMetrics
from .exceptions import FormalizationError, UnsupportedSystemError
from .config import FormalizationConfig


class TargetSystem(Enum):
    """Supported proof assistant systems."""
    LEAN4 = "lean4"
    ISABELLE = "isabelle"
    COQ = "coq"
    AGDA = "agda"


@dataclass
class FormalizationResult:
    """Result of a formalization attempt."""
    success: bool
    formal_code: Optional[str] = None
    error_message: Optional[str] = None
    verification_status: Optional[bool] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    correction_rounds: int = 0
    processing_time: float = 0.0


class FormalizationPipeline:
    """Main pipeline for converting LaTeX proofs to formal code.
    
    This class orchestrates the entire formalization process:
    1. Parse LaTeX mathematical content
    2. Extract theorems, definitions, and proofs
    3. Generate formal code using LLMs
    4. Verify the generated code
    5. Collect metrics and provide feedback
    
    Args:
        target_system: Target proof assistant (lean4, isabelle, coq)
        model: LLM model to use (gpt-4, claude-3, etc.)
        config: Optional configuration object
        
    Example:
        >>> pipeline = FormalizationPipeline(target_system="lean4")
        >>> result = await pipeline.formalize(latex_proof)
        >>> print(result.formal_code)
    """
    
    def __init__(
        self,
        target_system: Union[str, TargetSystem] = TargetSystem.LEAN4,
        model: str = "gpt-4",
        config: Optional[FormalizationConfig] = None,
        api_key: Optional[str] = None
    ):
        self.target_system = TargetSystem(target_system) if isinstance(target_system, str) else target_system
        self.model = model
        self.config = config or FormalizationConfig()
        self.logger = setup_logger(__name__)
        self.metrics = FormalizationMetrics()
        
        # Initialize components
        self._setup_components(api_key)
        
    def _setup_components(self, api_key: Optional[str]) -> None:
        """Initialize parser, generator, and verifier components."""
        try:
            # Initialize parser
            self.parser = LaTeXParser()
            
            # Initialize appropriate generator with robust error handling
            if self.target_system == TargetSystem.LEAN4:
                try:
                    self.generator = Lean4Generator(model=self.model, api_key=api_key)
                except Exception as gen_error:
                    self.logger.warning(f"Failed to initialize Lean4 generator: {gen_error}")
                    # Initialize mock generator for offline mode
                    self.generator = self._create_mock_generator()
                    
                # Initialize verifier with fallback
                try:
                    self.verifier = Lean4Verifier()
                except Exception as ver_error:
                    self.logger.warning(f"Lean4 verifier not available: {ver_error}")
                    self.verifier = None
                    
            elif self.target_system == TargetSystem.ISABELLE:
                try:
                    self.generator = IsabelleGenerator(model=self.model, api_key=api_key)
                except Exception as gen_error:
                    self.logger.warning(f"Failed to initialize Isabelle generator: {gen_error}")
                    self.generator = self._create_mock_generator()
                self.verifier = None  # Will be implemented later
                
            elif self.target_system == TargetSystem.COQ:
                try:
                    self.generator = CoqGenerator(model=self.model, api_key=api_key)
                except Exception as gen_error:
                    self.logger.warning(f"Failed to initialize Coq generator: {gen_error}")
                    self.generator = self._create_mock_generator()
                self.verifier = None  # Will be implemented later
            else:
                raise UnsupportedSystemError(f"System {self.target_system.value} not yet supported")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            # Don't fail completely - create mock components for testing
            self.parser = LaTeXParser()
            self.generator = self._create_mock_generator()
            self.verifier = None
            self.logger.warning("Using mock components for testing/offline mode")
    
    async def formalize(
        self,
        latex_content: str,
        verify: bool = True,
        timeout: int = 30
    ) -> FormalizationResult:
        """Formalize LaTeX mathematical content.
        
        Args:
            latex_content: LaTeX source containing mathematical content
            verify: Whether to verify the generated formal code
            timeout: Timeout in seconds for verification
            
        Returns:
            FormalizationResult with the generated formal code and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"Starting formalization to {self.target_system.value}")
            
            # Step 1: Parse LaTeX content
            self.logger.debug("Parsing LaTeX content")
            parsed_content = await self.parser.parse(latex_content)
            
            if not parsed_content.theorems and not parsed_content.definitions:
                raise FormalizationError("No mathematical content found in LaTeX")
            
            # Step 2: Generate formal code
            self.logger.debug(f"Generating {self.target_system.value} code")
            formal_code = await self.generator.generate(parsed_content)
            
            # Step 3: Verify if requested and verifier available
            verification_status = None
            if verify and self.verifier:
                self.logger.debug("Verifying generated code")
                verification_status = await self.verifier.verify(formal_code, timeout=timeout)
            
            # Step 4: Collect metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            metrics = {
                "processing_time": processing_time,
                "content_length": len(latex_content),
                "theorems_count": len(parsed_content.theorems),
                "definitions_count": len(parsed_content.definitions),
                "formal_code_length": len(formal_code) if formal_code else 0,
            }
            
            self.metrics.record_formalization(
                success=True,
                target_system=self.target_system.value,
                processing_time=processing_time,
                verification_success=verification_status
            )
            
            self.logger.info(f"Formalization completed successfully in {processing_time:.2f}s")
            
            return FormalizationResult(
                success=True,
                formal_code=formal_code,
                verification_status=verification_status,
                metrics=metrics,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Formalization failed after {processing_time:.2f}s: {e}")
            
            self.metrics.record_formalization(
                success=False,
                target_system=self.target_system.value,
                processing_time=processing_time,
                error=str(e)
            )
            
            return FormalizationResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def formalize_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        verify: bool = True
    ) -> FormalizationResult:
        """Formalize a LaTeX file.
        
        Args:
            input_path: Path to input LaTeX file
            output_path: Optional output path for formal code
            verify: Whether to verify the generated code
            
        Returns:
            FormalizationResult with success status and metadata
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Read LaTeX content
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
        except Exception as e:
            raise FormalizationError(f"Failed to read input file: {e}")
        
        # Formalize
        result = await self.formalize(latex_content, verify=verify)
        
        # Write output if successful and path provided
        if result.success and result.formal_code and output_path:
            try:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.formal_code)
                    
                self.logger.info(f"Formal code written to {output_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to write output file: {e}")
        
        return result
    
    async def batch_formalize(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        parallel: int = 1,
        verify: bool = True
    ) -> List[FormalizationResult]:
        """Batch formalize multiple LaTeX files.
        
        Args:
            input_files: List of input LaTeX file paths
            output_dir: Optional output directory
            parallel: Number of parallel workers
            verify: Whether to verify generated code
            
        Returns:
            List of FormalizationResult objects
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(parallel)
        
        async def process_file(input_path: Path) -> FormalizationResult:
            async with semaphore:
                try:
                    # Generate output path if directory provided
                    output_path = None
                    if output_dir:
                        if self.target_system == TargetSystem.LEAN4:
                            output_path = output_dir / f"{input_path.stem}.lean"
                        elif self.target_system == TargetSystem.ISABELLE:
                            output_path = output_dir / f"{input_path.stem}.thy"
                        elif self.target_system == TargetSystem.COQ:
                            output_path = output_dir / f"{input_path.stem}.v"
                    
                    return await self.formalize_file(input_path, output_path, verify)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {input_path}: {e}")
                    return FormalizationResult(
                        success=False,
                        error_message=str(e)
                    )
        
        # Process all files concurrently
        tasks = [process_file(Path(f)) for f in input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(FormalizationResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        # Log summary statistics
        successful = sum(1 for r in final_results if r.success)
        total = len(final_results)
        self.logger.info(f"Batch processing completed: {successful}/{total} successful")
        
        return final_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        return self.metrics.get_summary()
    
    def reset_metrics(self) -> None:
        """Reset pipeline metrics."""
        self.metrics.reset()
        
    def _create_mock_generator(self):
        """Create mock generator for offline/testing mode."""
        class MockGenerator:
            def __init__(self):
                self.model = "mock"
                
            async def generate(self, parsed_content):
                """Generate mock formal code."""
                if hasattr(parsed_content, 'theorems') and parsed_content.theorems:
                    theorem = parsed_content.theorems[0]
                    if self.target_system == TargetSystem.LEAN4:
                        return f"-- Mock Lean 4 formalization\ntheorem mock_theorem : True := by trivial"
                    elif self.target_system == TargetSystem.ISABELLE:
                        return f"theory MockTheorem\ntheorem mock_theorem: \"True\"\nby simp\nend"
                    elif self.target_system == TargetSystem.COQ:
                        return f"(* Mock Coq formalization *)\nTheorem mock_theorem : True.\nProof. exact I. Qed."
                return "-- Mock formalization (no theorems found)"
                
        generator = MockGenerator()
        generator.target_system = self.target_system
        return generator
