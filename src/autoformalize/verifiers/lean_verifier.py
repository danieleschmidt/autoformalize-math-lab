"""Lean 4 proof verification.

This module provides functionality to verify generated Lean 4 proofs
by interfacing with the Lean 4 compiler and proof checker.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..core.exceptions import VerificationError, TimeoutError
from ..utils.logging_config import setup_logger


@dataclass
class VerificationResult:
    """Result of Lean 4 proof verification."""
    success: bool
    output: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class Lean4Verifier:
    """Verifier for Lean 4 formal proofs.
    
    This class provides functionality to verify generated Lean 4 code
    by running it through the Lean 4 compiler and proof checker.
    """
    
    def __init__(
        self,
        lean_executable: str = "lean",
        mathlib_path: Optional[Path] = None,
        timeout: int = 30
    ):
        """Initialize the Lean 4 verifier.
        
        Args:
            lean_executable: Path to the Lean 4 executable
            mathlib_path: Path to Mathlib installation
            timeout: Default timeout in seconds
        """
        self.lean_executable = lean_executable
        self.mathlib_path = mathlib_path
        self.timeout = timeout
        self.logger = setup_logger(__name__)
    
    async def verify(self, lean_code: str, timeout: Optional[int] = None) -> bool:
        """Verify Lean 4 code.
        
        Args:
            lean_code: Lean 4 code to verify
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            True if verification succeeds, False otherwise
        """
        result = await self.verify_detailed(lean_code, timeout)
        return result.success
    
    async def verify_detailed(
        self,
        lean_code: str,
        timeout: Optional[int] = None
    ) -> VerificationResult:
        """Verify Lean 4 code with detailed results.
        
        Args:
            lean_code: Lean 4 code to verify
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            VerificationResult with detailed information
        """
        if timeout is None:
            timeout = self.timeout
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.debug("Starting Lean 4 verification")
            
            # Create temporary file for the Lean code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.lean',
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                tmp_file.write(lean_code)
                tmp_path = Path(tmp_file.name)
            
            try:
                # Run Lean 4 compiler
                result = await self._run_lean_compiler(tmp_path, timeout)
                
                processing_time = asyncio.get_event_loop().time() - start_time
                result.processing_time = processing_time
                
                if result.success:
                    self.logger.info(f"Lean verification succeeded in {processing_time:.2f}s")
                else:
                    self.logger.warning(f"Lean verification failed in {processing_time:.2f}s")
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    tmp_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file: {e}")
        
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Verification failed: {e}")
            
            return VerificationResult(
                success=False,
                output="",
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def _run_lean_compiler(
        self,
        lean_file: Path,
        timeout: int
    ) -> VerificationResult:
        """Run the Lean 4 compiler on a file.
        
        Args:
            lean_file: Path to the Lean file
            timeout: Timeout in seconds
            
        Returns:
            VerificationResult with compiler output
        """
        try:
            # Prepare command
            cmd = [self.lean_executable, str(lean_file)]
            
            # Add Mathlib path if specified
            if self.mathlib_path:
                cmd.extend(["--path", str(self.mathlib_path)])
            
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Run the compiler
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=lean_file.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Lean verification timed out after {timeout} seconds")
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            # Parse output
            success = process.returncode == 0
            output = stdout_text + stderr_text
            
            # Extract errors and warnings
            errors = self._extract_errors(output)
            warnings = self._extract_warnings(output)
            
            return VerificationResult(
                success=success,
                output=output,
                errors=errors,
                warnings=warnings
            )
            
        except FileNotFoundError:
            raise VerificationError(
                f"Lean executable not found: {self.lean_executable}. "
                "Please install Lean 4 or set the correct path."
            )
        except Exception as e:
            raise VerificationError(f"Failed to run Lean compiler: {e}")
    
    def _extract_errors(self, output: str) -> List[str]:
        """Extract error messages from Lean output.
        
        Args:
            output: Lean compiler output
            
        Returns:
            List of error messages
        """
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            # Look for error patterns
            if 'error:' in line.lower() or 'failed' in line.lower():
                errors.append(line.strip())
            elif line.startswith('│') and 'error' in line:
                errors.append(line.strip())
        
        return errors
    
    def _extract_warnings(self, output: str) -> List[str]:
        """Extract warning messages from Lean output.
        
        Args:
            output: Lean compiler output
            
        Returns:
            List of warning messages
        """
        warnings = []
        lines = output.split('\n')
        
        for line in lines:
            # Look for warning patterns
            if 'warning:' in line.lower():
                warnings.append(line.strip())
            elif line.startswith('│') and 'warning' in line:
                warnings.append(line.strip())
        
        return warnings
    
    async def check_lean_installation(self) -> Dict[str, Any]:
        """Check if Lean 4 is properly installed.
        
        Returns:
            Dictionary with installation status and version info
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.lean_executable, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_info = stdout.decode('utf-8', errors='replace').strip()
                return {
                    "installed": True,
                    "version": version_info,
                    "executable": self.lean_executable
                }
            else:
                error_info = stderr.decode('utf-8', errors='replace').strip()
                return {
                    "installed": False,
                    "error": error_info,
                    "executable": self.lean_executable
                }
                
        except FileNotFoundError:
            return {
                "installed": False,
                "error": f"Executable not found: {self.lean_executable}",
                "executable": self.lean_executable
            }
        except Exception as e:
            return {
                "installed": False,
                "error": str(e),
                "executable": self.lean_executable
            }
    
    async def validate_mathlib(self) -> Dict[str, Any]:
        """Validate Mathlib installation.
        
        Returns:
            Dictionary with Mathlib validation results
        """
        if not self.mathlib_path:
            return {
                "available": False,
                "message": "No Mathlib path configured"
            }
        
        try:
            # Create a simple test that uses Mathlib
            test_code = """import Mathlib.Data.Nat.Basic

example : ∀ n : ℕ, n + 0 = n := fun n => Nat.add_zero n
"""
            
            result = await self.verify_detailed(test_code)
            
            return {
                "available": result.success,
                "path": str(self.mathlib_path),
                "test_result": {
                    "success": result.success,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            }
            
        except Exception as e:
            return {
                "available": False,
                "path": str(self.mathlib_path),
                "error": str(e)
            }
    
    def set_mathlib_path(self, path: Path) -> None:
        """Set the Mathlib installation path.
        
        Args:
            path: Path to Mathlib installation
        """
        self.mathlib_path = path
        self.logger.info(f"Set Mathlib path to: {path}")
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        # This would be expanded to track actual verification statistics
        return {
            "lean_executable": self.lean_executable,
            "mathlib_path": str(self.mathlib_path) if self.mathlib_path else None,
            "default_timeout": self.timeout
        }
