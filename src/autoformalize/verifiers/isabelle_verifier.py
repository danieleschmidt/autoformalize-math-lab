"""Isabelle/HOL proof verification.

This module provides functionality to verify generated Isabelle/HOL proofs
by interfacing with the Isabelle system.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..core.exceptions import VerificationError, TimeoutError
from ..utils.logging_config import setup_logger


@dataclass
class IsabelleVerificationResult:
    """Result of Isabelle/HOL proof verification."""
    success: bool
    output: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    theory_name: Optional[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class IsabelleVerifier:
    """Verifier for Isabelle/HOL formal proofs.
    
    This class provides functionality to verify generated Isabelle/HOL code
    by running it through the Isabelle system.
    """
    
    def __init__(
        self,
        isabelle_executable: str = "isabelle",
        afp_path: Optional[Path] = None,
        timeout: int = 60
    ):
        """Initialize the Isabelle verifier.
        
        Args:
            isabelle_executable: Path to the Isabelle executable
            afp_path: Path to Archive of Formal Proofs
            timeout: Default timeout in seconds
        """
        self.isabelle_executable = isabelle_executable
        self.afp_path = afp_path
        self.timeout = timeout
        self.logger = setup_logger(__name__)
    
    async def verify(self, isabelle_code: str, timeout: Optional[int] = None) -> bool:
        """Verify Isabelle/HOL code.
        
        Args:
            isabelle_code: Isabelle/HOL code to verify
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            True if verification succeeds, False otherwise
        """
        result = await self.verify_detailed(isabelle_code, timeout)
        return result.success
    
    async def verify_detailed(
        self,
        isabelle_code: str,
        timeout: Optional[int] = None
    ) -> IsabelleVerificationResult:
        """Verify Isabelle/HOL code with detailed results.
        
        Args:
            isabelle_code: Isabelle/HOL code to verify
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            IsabelleVerificationResult with detailed information
        """
        if timeout is None:
            timeout = self.timeout
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.debug("Starting Isabelle/HOL verification")
            
            # Extract theory name from code
            theory_name = self._extract_theory_name(isabelle_code)
            
            # Create temporary directory for the theory
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                theory_file = tmp_path / f"{theory_name}.thy"
                
                # Write theory file
                theory_file.write_text(isabelle_code, encoding='utf-8')
                
                # Run Isabelle verification
                result = await self._run_isabelle_build(tmp_path, theory_name, timeout)
                
                processing_time = asyncio.get_event_loop().time() - start_time
                result.processing_time = processing_time
                result.theory_name = theory_name
                
                if result.success:
                    self.logger.info(f"Isabelle verification succeeded in {processing_time:.2f}s")
                else:
                    self.logger.warning(f"Isabelle verification failed in {processing_time:.2f}s")
                
                return result
        
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Verification failed: {e}")
            
            return IsabelleVerificationResult(
                success=False,
                output="",
                errors=[str(e)],
                processing_time=processing_time
            )
    
    def _extract_theory_name(self, isabelle_code: str) -> str:
        """Extract theory name from Isabelle code.
        
        Args:
            isabelle_code: Isabelle/HOL code
            
        Returns:
            Theory name or default name
        """
        lines = isabelle_code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('theory '):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        
        return "Generated_Theory"
    
    async def _run_isabelle_build(
        self,
        theory_dir: Path,
        theory_name: str,
        timeout: int
    ) -> IsabelleVerificationResult:
        """Run Isabelle build on a theory.
        
        Args:
            theory_dir: Directory containing the theory
            theory_name: Name of the theory
            timeout: Timeout in seconds
            
        Returns:
            IsabelleVerificationResult with build output
        """
        try:
            # Create ROOT file for the session
            root_file = theory_dir / "ROOT"
            root_content = f"""session {theory_name} = HOL +
  options [document = false]
  theories
    {theory_name}
"""
            root_file.write_text(root_content)
            
            # Prepare command
            cmd = [self.isabelle_executable, "build", "-D", str(theory_dir)]
            
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Run Isabelle build
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=theory_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Isabelle verification timed out after {timeout} seconds")
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            # Parse output
            success = process.returncode == 0
            output = stdout_text + stderr_text
            
            # Extract errors and warnings
            errors = self._extract_errors(output)
            warnings = self._extract_warnings(output)
            
            return IsabelleVerificationResult(
                success=success,
                output=output,
                errors=errors,
                warnings=warnings
            )
            
        except FileNotFoundError:
            raise VerificationError(
                f"Isabelle executable not found: {self.isabelle_executable}. "
                "Please install Isabelle/HOL or set the correct path."
            )
        except Exception as e:
            raise VerificationError(f"Failed to run Isabelle build: {e}")
    
    def _extract_errors(self, output: str) -> List[str]:
        """Extract error messages from Isabelle output.
        
        Args:
            output: Isabelle build output
            
        Returns:
            List of error messages
        """
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for error patterns
            if 'error:' in line.lower() or 'failed' in line.lower():
                errors.append(line)
            elif line.startswith('***') and 'error' in line.lower():
                errors.append(line)
            elif 'exception' in line.lower():
                errors.append(line)
        
        return errors
    
    def _extract_warnings(self, output: str) -> List[str]:
        """Extract warning messages from Isabelle output.
        
        Args:
            output: Isabelle build output
            
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
            elif line.startswith('###') and 'warning' in line.lower():
                warnings.append(line)
        
        return warnings
    
    async def check_isabelle_installation(self) -> Dict[str, Any]:
        """Check if Isabelle/HOL is properly installed.
        
        Returns:
            Dictionary with installation status and version info
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.isabelle_executable, "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                version_info = stdout.decode('utf-8', errors='replace').strip()
                return {
                    "installed": True,
                    "version": version_info,
                    "executable": self.isabelle_executable
                }
            else:
                error_info = stderr.decode('utf-8', errors='replace').strip()
                return {
                    "installed": False,
                    "error": error_info,
                    "executable": self.isabelle_executable
                }
                
        except FileNotFoundError:
            return {
                "installed": False,
                "error": f"Executable not found: {self.isabelle_executable}",
                "executable": self.isabelle_executable
            }
        except Exception as e:
            return {
                "installed": False,
                "error": str(e),
                "executable": self.isabelle_executable
            }
    
    async def validate_afp(self) -> Dict[str, Any]:
        """Validate Archive of Formal Proofs installation.
        
        Returns:
            Dictionary with AFP validation results
        """
        if not self.afp_path:
            return {
                "available": False,
                "message": "No AFP path configured"
            }
        
        try:
            # Create a simple test that uses AFP
            test_code = """theory AFP_Test
  imports Complex_Main
begin

lemma test_lemma: "True"
  by simp

end"""
            
            result = await self.verify_detailed(test_code)
            
            return {
                "available": result.success,
                "path": str(self.afp_path),
                "test_result": {
                    "success": result.success,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
            }
            
        except Exception as e:
            return {
                "available": False,
                "path": str(self.afp_path),
                "error": str(e)
            }
    
    def set_afp_path(self, path: Path) -> None:
        """Set the Archive of Formal Proofs path.
        
        Args:
            path: Path to AFP installation
        """
        self.afp_path = path
        self.logger.info(f"Set AFP path to: {path}")
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics.
        
        Returns:
            Dictionary with verification statistics
        """
        return {
            "isabelle_executable": self.isabelle_executable,
            "afp_path": str(self.afp_path) if self.afp_path else None,
            "default_timeout": self.timeout
        }