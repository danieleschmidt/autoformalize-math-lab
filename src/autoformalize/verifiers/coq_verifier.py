"""Coq proof verification.

This module provides functionality to verify generated Coq proofs
by interfacing with the Coq proof assistant.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..core.exceptions import VerificationError, TimeoutError
from ..utils.logging_config import setup_logger


@dataclass
class CoqVerificationResult:
    """Result of Coq proof verification."""
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


class CoqVerifier:
    """Verifier for Coq formal proofs.
    
    This class provides functionality to verify generated Coq code
    by running it through the Coq compiler.
    """
    
    def __init__(
        self,
        coq_executable: str = "coqc",
        coq_lib_path: Optional[Path] = None,
        timeout: int = 30
    ):
        """Initialize the Coq verifier.
        
        Args:
            coq_executable: Path to the Coq compiler
            coq_lib_path: Path to Coq standard library
            timeout: Default timeout in seconds
        """
        self.coq_executable = coq_executable
        self.coq_lib_path = coq_lib_path
        self.timeout = timeout
        self.logger = setup_logger(__name__)
    
    async def verify(self, coq_code: str, timeout: Optional[int] = None) -> bool:
        """Verify Coq code.
        
        Args:
            coq_code: Coq code to verify
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            True if verification succeeds, False otherwise
        """
        result = await self.verify_detailed(coq_code, timeout)
        return result.success
    
    async def verify_detailed(
        self,
        coq_code: str,
        timeout: Optional[int] = None
    ) -> CoqVerificationResult:
        """Verify Coq code with detailed results.
        
        Args:
            coq_code: Coq code to verify
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            CoqVerificationResult with detailed information
        """
        if timeout is None:
            timeout = self.timeout
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.debug("Starting Coq verification")
            
            # Create temporary file for the Coq code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.v',
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                tmp_file.write(coq_code)
                tmp_path = Path(tmp_file.name)
            
            try:
                # Run Coq compiler
                result = await self._run_coq_compiler(tmp_path, timeout)
                
                processing_time = asyncio.get_event_loop().time() - start_time
                result.processing_time = processing_time
                
                if result.success:
                    self.logger.info(f"Coq verification succeeded in {processing_time:.2f}s")
                else:
                    self.logger.warning(f"Coq verification failed in {processing_time:.2f}s")
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    tmp_path.unlink()
                    # Also clean up .vo file if created
                    vo_file = tmp_path.with_suffix('.vo')
                    if vo_file.exists():
                        vo_file.unlink()
                    # Clean up .vok file if created  
                    vok_file = tmp_path.with_suffix('.vok')
                    if vok_file.exists():
                        vok_file.unlink()
                    # Clean up .vos file if created
                    vos_file = tmp_path.with_suffix('.vos')
                    if vos_file.exists():
                        vos_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary files: {e}")
        
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Verification failed: {e}")
            
            return CoqVerificationResult(
                success=False,
                output="",
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def _run_coq_compiler(
        self,
        coq_file: Path,
        timeout: int
    ) -> CoqVerificationResult:
        """Run the Coq compiler on a file.
        
        Args:
            coq_file: Path to the Coq file
            timeout: Timeout in seconds
            
        Returns:
            CoqVerificationResult with compiler output
        """
        try:
            # Prepare command
            cmd = [self.coq_executable, str(coq_file)]
            
            # Add library path if specified
            if self.coq_lib_path:
                cmd.extend(["-I", str(self.coq_lib_path)])
            
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Run the compiler
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=coq_file.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Coq verification timed out after {timeout} seconds")
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            # Parse output
            success = process.returncode == 0
            output = stdout_text + stderr_text
            
            # Extract errors and warnings
            errors = self._extract_errors(output)
            warnings = self._extract_warnings(output)
            
            return CoqVerificationResult(
                success=success,
                output=output,
                errors=errors,
                warnings=warnings
            )
            
        except FileNotFoundError:
            raise VerificationError(
                f"Coq executable not found: {self.coq_executable}. "
                "Please install Coq or set the correct path."
            )
        except Exception as e:
            raise VerificationError(f"Failed to run Coq compiler: {e}")
    
    def _extract_errors(self, output: str) -> List[str]:
        """Extract error messages from Coq output.
        
        Args:
            output: Coq compiler output
            
        Returns:
            List of error messages
        """
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for error patterns
            if 'error:' in line.lower() or 'error' in line.lower():
                errors.append(line)
            elif line.startswith('File') and 'Error:' in line:
                errors.append(line)
            elif 'syntax error' in line.lower():
                errors.append(line)
            elif 'anomaly' in line.lower():
                errors.append(line)
        
        return errors
    
    def _extract_warnings(self, output: str) -> List[str]:
        """Extract warning messages from Coq output.
        
        Args:
            output: Coq compiler output
            
        Returns:
            List of warning messages
        """
        warnings = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for warning patterns
            if 'warning:' in line.lower():
                warnings.append(line)
            elif line.startswith('Warning:'):
                warnings.append(line)
        
        return warnings
    
    async def check_coq_installation(self) -> Dict[str, Any]:
        """Check if Coq is properly installed.
        
        Returns:
            Dictionary with installation status and version info
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.coq_executable, "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_info = stdout.decode('utf-8', errors='replace').strip()
                return {
                    "installed": True,
                    "version": version_info,
                    "executable": self.coq_executable
                }
            else:
                error_info = stderr.decode('utf-8', errors='replace').strip()
                return {
                    "installed": False,
                    "error": error_info,
                    "executable": self.coq_executable
                }
                
        except FileNotFoundError:
            return {
                "installed": False,
                "error": f"Executable not found: {self.coq_executable}",
                "executable": self.coq_executable
            }
        except Exception as e:
            return {
                "installed": False,
                "error": str(e),
                "executable": self.coq_executable
            }
    
    async def validate_standard_library(self) -> Dict[str, Any]:
        """Validate Coq standard library installation.
        
        Returns:
            Dictionary with standard library validation results
        """
        try:
            # Create a simple test that uses standard library
            test_code = """Require Import Arith.
Require Import Logic.

Example test_example : forall n : nat, n + 0 = n.
Proof.
  intro n.
  apply Nat.add_0_r.
Qed.
"""
            
            result = await self.verify_detailed(test_code)
            
            return {
                "available": result.success,
                "lib_path": str(self.coq_lib_path) if self.coq_lib_path else "default",
                "test_result": {
                    "success": result.success,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            }
            
        except Exception as e:
            return {
                "available": False,
                "lib_path": str(self.coq_lib_path) if self.coq_lib_path else "default",
                "error": str(e)
            }
    
    async def check_interactive_mode(self) -> Dict[str, Any]:
        """Check if Coq interactive mode (coqtop) is available.
        
        Returns:
            Dictionary with interactive mode availability
        """
        try:
            # Try to run coqtop
            coqtop_executable = self.coq_executable.replace('coqc', 'coqtop')
            
            process = await asyncio.create_subprocess_exec(
                coqtop_executable, "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=5
            )
            
            if process.returncode == 0:
                return {
                    "available": True,
                    "executable": coqtop_executable,
                    "version": stdout.decode('utf-8', errors='replace').strip()
                }
            else:
                return {
                    "available": False,
                    "executable": coqtop_executable,
                    "error": stderr.decode('utf-8', errors='replace').strip()
                }
                
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def set_lib_path(self, path: Path) -> None:
        """Set the Coq library path.
        
        Args:
            path: Path to Coq library installation
        """
        self.coq_lib_path = path
        self.logger.info(f"Set Coq library path to: {path}")
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        return {
            "coq_executable": self.coq_executable,
            "coq_lib_path": str(self.coq_lib_path) if self.coq_lib_path else None,
            "default_timeout": self.timeout
        }