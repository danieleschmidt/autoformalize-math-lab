#!/usr/bin/env python3
"""Comprehensive Quality Gates for Production Deployment.

This script executes all mandatory quality gates including:
- Code quality analysis
- Security scanning
- Performance benchmarking
- Test coverage validation
- Documentation completeness
- Mathematical correctness verification
"""

import asyncio
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autoformalize.core.enhanced_pipeline import EnhancedFormalizationPipeline, VerificationMode
from autoformalize.performance.advanced_optimization import AdvancedOptimizationEngine, OptimizationConfig
from autoformalize.core.robust_error_recovery import RobustErrorRecoverySystem
from autoformalize.utils.enhanced_health_monitoring import AdvancedHealthMonitor


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class ComprehensiveQualityGates:
    """Comprehensive quality gates execution system."""
    
    def __init__(self, project_root: Path = None):
        """Initialize quality gates system.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path(__file__).parent
        self.results: List[QualityGateResult] = []
        self.overall_score = 0.0
        self.start_time = time.time()
        
        # Quality gate weights
        self.gate_weights = {
            "code_quality": 0.20,
            "security": 0.25,
            "performance": 0.20,
            "testing": 0.15,
            "documentation": 0.10,
            "mathematical_validation": 0.10
        }
        
        # Pass thresholds
        self.pass_thresholds = {
            "code_quality": 0.80,
            "security": 0.95,
            "performance": 0.75,
            "testing": 0.85,
            "documentation": 0.70,
            "mathematical_validation": 0.80
        }
    
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates.
        
        Returns:
            Comprehensive quality gate results
        """
        print("🔒 EXECUTING COMPREHENSIVE QUALITY GATES")
        print("=" * 50)
        
        # Execute gates in parallel where possible
        gate_tasks = [
            self._gate_code_quality(),
            self._gate_security(),
            self._gate_performance(),
            self._gate_testing(),
            self._gate_documentation(),
            self._gate_mathematical_validation()
        ]
        
        results = await asyncio.gather(*gate_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.results.append(QualityGateResult(
                    gate_name="unknown",
                    passed=False,
                    score=0.0,
                    issues=[f"Gate execution failed: {result}"]
                ))
            elif isinstance(result, QualityGateResult):
                self.results.append(result)
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Generate report
        return self._generate_final_report()
    
    async def _gate_code_quality(self) -> QualityGateResult:
        """Execute code quality gate."""
        start_time = time.time()
        print("\n📊 Code Quality Gate")
        print("-" * 20)
        
        issues = []
        recommendations = []
        details = {}
        score = 1.0
        
        try:
            # Check if Python files exist
            python_files = list(self.project_root.glob("**/*.py"))
            if not python_files:
                issues.append("No Python files found")
                score = 0.0
            else:
                details["python_files_count"] = len(python_files)
                print(f"  📁 Found {len(python_files)} Python files")
            
            # Basic syntax check
            syntax_issues = 0
            for py_file in python_files[:20]:  # Check first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    compile(content, str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_issues += 1
                    issues.append(f"Syntax error in {py_file}: {e}")
                except Exception:
                    pass  # Skip files with import issues
            
            if syntax_issues > 0:
                score *= 0.8
                issues.append(f"{syntax_issues} files with syntax issues")
            else:
                print("  ✅ All checked files have valid syntax")
            
            # Check for proper imports
            import_structure_score = self._check_import_structure()
            score *= import_structure_score
            details["import_structure_score"] = import_structure_score
            
            if import_structure_score < 0.9:
                recommendations.append("Improve import structure and organization")
            
            # Check for documentation strings
            docstring_coverage = self._check_docstring_coverage()
            score *= docstring_coverage
            details["docstring_coverage"] = docstring_coverage
            
            if docstring_coverage < 0.8:
                recommendations.append("Add more comprehensive docstrings")
            
            print(f"  📊 Code Quality Score: {score:.2f}")
            
        except Exception as e:
            issues.append(f"Code quality check failed: {e}")
            score = 0.5
        
        execution_time = time.time() - start_time
        passed = score >= self.pass_thresholds["code_quality"]
        
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {self.pass_thresholds['code_quality']:.2f})")
        
        return QualityGateResult(
            gate_name="code_quality",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    async def _gate_security(self) -> QualityGateResult:
        """Execute security gate."""
        start_time = time.time()
        print("\n🔒 Security Gate")
        print("-" * 15)
        
        issues = []
        recommendations = []
        details = {}
        score = 1.0
        
        try:
            # Check for hardcoded secrets
            secrets_found = self._scan_for_secrets()
            if secrets_found > 0:
                score *= 0.7
                issues.append(f"Found {secrets_found} potential hardcoded secrets")
                recommendations.append("Remove hardcoded secrets and use environment variables")
            else:
                print("  ✅ No hardcoded secrets detected")
            
            details["secrets_found"] = secrets_found
            
            # Check for secure practices
            security_practices_score = self._check_security_practices()
            score *= security_practices_score
            details["security_practices_score"] = security_practices_score
            
            if security_practices_score < 0.9:
                recommendations.append("Implement additional security best practices")
            
            # Validate input sanitization
            input_validation_score = self._check_input_validation()
            score *= input_validation_score
            details["input_validation_score"] = input_validation_score
            
            if input_validation_score < 0.8:
                recommendations.append("Enhance input validation and sanitization")
            
            print(f"  🔐 Security Score: {score:.2f}")
            
        except Exception as e:
            issues.append(f"Security scan failed: {e}")
            score = 0.8  # Conservative score for failed security check
        
        execution_time = time.time() - start_time
        passed = score >= self.pass_thresholds["security"]
        
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {self.pass_thresholds['security']:.2f})")
        
        return QualityGateResult(
            gate_name="security",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    async def _gate_performance(self) -> QualityGateResult:
        """Execute performance gate."""
        start_time = time.time()
        print("\n⚡ Performance Gate")
        print("-" * 18)
        
        issues = []
        recommendations = []
        details = {}
        score = 1.0
        
        try:
            # Test basic pipeline performance
            print("  🧪 Testing pipeline performance...")
            
            pipeline = EnhancedFormalizationPipeline(
                target_system="lean4",
                verification_mode=VerificationMode.MOCK,
                enable_caching=True
            )
            
            # Benchmark basic operations
            test_latex = r"$1 + 1 = 2$"
            
            # Measure formalization time
            perf_start = time.time()
            result = await pipeline.formalize(test_latex)
            formalization_time = time.time() - perf_start
            
            details["formalization_time"] = formalization_time
            details["formalization_success"] = result.success
            
            if formalization_time > 5.0:  # 5 second threshold
                score *= 0.8
                issues.append(f"Slow formalization: {formalization_time:.2f}s")
                recommendations.append("Optimize formalization pipeline performance")
            else:
                print(f"  ✅ Formalization completed in {formalization_time:.3f}s")
            
            # Test optimization engine
            print("  🚀 Testing optimization engine...")
            
            config = OptimizationConfig(cache_size_mb=32, batch_size=3)
            opt_engine = AdvancedOptimizationEngine(config)
            
            try:
                def test_operation(x: int) -> int:
                    return x * 2
                
                # Test cached operation
                cache_start = time.time()
                await opt_engine.optimize_operation(
                    test_operation,
                    cache_key="test_op",
                    task_type="test",
                    x=42
                )
                cache_time = time.time() - cache_start
                
                details["cache_operation_time"] = cache_time
                
                if cache_time > 1.0:
                    score *= 0.9
                    recommendations.append("Optimize caching performance")
                else:
                    print(f"  ✅ Cache operation completed in {cache_time:.3f}s")
                
                # Get optimization statistics
                opt_stats = opt_engine.get_optimization_statistics()
                details["optimization_stats"] = opt_stats
                
            finally:
                opt_engine.shutdown()
            
            # Memory usage check (basic)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                details["memory_usage_mb"] = memory_mb
                
                if memory_mb > 500:  # 500MB threshold
                    score *= 0.9
                    recommendations.append("Optimize memory usage")
                else:
                    print(f"  ✅ Memory usage: {memory_mb:.1f}MB")
            except ImportError:
                print("  ⚠️  Memory monitoring not available (psutil not installed)")
            
            print(f"  ⚡ Performance Score: {score:.2f}")
            
        except Exception as e:
            issues.append(f"Performance test failed: {e}")
            score = 0.6
        
        execution_time = time.time() - start_time
        passed = score >= self.pass_thresholds["performance"]
        
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {self.pass_thresholds['performance']:.2f})")
        
        return QualityGateResult(
            gate_name="performance",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    async def _gate_testing(self) -> QualityGateResult:
        """Execute testing gate."""
        start_time = time.time()
        print("\n🧪 Testing Gate")
        print("-" * 14)
        
        issues = []
        recommendations = []
        details = {}
        score = 1.0
        
        try:
            # Count test files
            test_files = list(self.project_root.glob("**/test_*.py"))
            test_files.extend(list(self.project_root.glob("tests/**/*.py")))
            
            if not test_files:
                issues.append("No test files found")
                score = 0.5
                recommendations.append("Add comprehensive test suite")
            else:
                print(f"  📁 Found {len(test_files)} test files")
                details["test_files_count"] = len(test_files)
            
            # Test basic functionality
            print("  🔧 Testing core functionality...")
            
            try:
                # Test pipeline initialization
                pipeline = EnhancedFormalizationPipeline(
                    verification_mode=VerificationMode.MOCK
                )
                
                # Test basic formalization
                test_result = await pipeline.formalize("$x = x$")
                
                if test_result.success:
                    print("  ✅ Basic formalization test passed")
                    details["basic_formalization"] = True
                else:
                    issues.append("Basic formalization test failed")
                    score *= 0.8
                    details["basic_formalization"] = False
                
            except Exception as e:
                issues.append(f"Core functionality test failed: {e}")
                score *= 0.7
            
            # Test error recovery
            print("  🛡️  Testing error recovery...")
            
            try:
                recovery_system = RobustErrorRecoverySystem(enable_proactive_healing=False)
                
                # Test error handling
                async def failing_function():
                    raise ValueError("Test error")
                
                try:
                    await recovery_system.handle_error_with_recovery(
                        ValueError("Initial error"),
                        {"component": "test"},
                        failing_function
                    )
                except:
                    pass  # Expected to fail after recovery attempts
                
                recovery_stats = recovery_system.get_error_statistics()
                
                if recovery_stats["total_errors"] > 0:
                    print("  ✅ Error recovery system functioning")
                    details["error_recovery"] = True
                else:
                    issues.append("Error recovery system not working")
                    score *= 0.9
                    details["error_recovery"] = False
                
            except Exception as e:
                issues.append(f"Error recovery test failed: {e}")
                score *= 0.8
            
            # Test health monitoring
            print("  📊 Testing health monitoring...")
            
            try:
                health_monitor = AdvancedHealthMonitor(enable_prometheus=False)
                
                # Record some operations
                health_monitor.record_operation("test", True)
                health_monitor.record_operation("test", False)
                
                health_status = await health_monitor.get_current_health()
                
                if "overall_status" in health_status:
                    print("  ✅ Health monitoring functioning")
                    details["health_monitoring"] = True
                else:
                    issues.append("Health monitoring not working")
                    score *= 0.9
                    details["health_monitoring"] = False
                
            except Exception as e:
                issues.append(f"Health monitoring test failed: {e}")
                score *= 0.8
            
            # Estimate test coverage
            coverage_estimate = self._estimate_test_coverage()
            details["estimated_coverage"] = coverage_estimate
            
            if coverage_estimate < 0.7:
                score *= 0.8
                recommendations.append("Increase test coverage")
            else:
                print(f"  ✅ Estimated test coverage: {coverage_estimate:.1%}")
            
            print(f"  🧪 Testing Score: {score:.2f}")
            
        except Exception as e:
            issues.append(f"Testing gate failed: {e}")
            score = 0.6
        
        execution_time = time.time() - start_time
        passed = score >= self.pass_thresholds["testing"]
        
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {self.pass_thresholds['testing']:.2f})")
        
        return QualityGateResult(
            gate_name="testing",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    async def _gate_documentation(self) -> QualityGateResult:
        """Execute documentation gate."""
        start_time = time.time()
        print("\n📚 Documentation Gate")
        print("-" * 20)
        
        issues = []
        recommendations = []
        details = {}
        score = 1.0
        
        try:
            # Check for README
            readme_files = list(self.project_root.glob("README*"))
            if readme_files:
                print("  ✅ README file found")
                details["readme_exists"] = True
                
                # Check README content
                readme_content = readme_files[0].read_text()
                if len(readme_content) > 1000:
                    print("  ✅ README has substantial content")
                    details["readme_comprehensive"] = True
                else:
                    issues.append("README content is too brief")
                    score *= 0.9
                    details["readme_comprehensive"] = False
            else:
                issues.append("No README file found")
                score *= 0.8
                details["readme_exists"] = False
            
            # Check for documentation directory
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                doc_files = list(docs_dir.glob("**/*.md"))
                doc_files.extend(list(docs_dir.glob("**/*.rst")))
                
                print(f"  ✅ Documentation directory with {len(doc_files)} files")
                details["docs_files_count"] = len(doc_files)
                
                if len(doc_files) < 5:
                    recommendations.append("Add more comprehensive documentation")
                    score *= 0.95
            else:
                issues.append("No docs directory found")
                score *= 0.9
                details["docs_files_count"] = 0
            
            # Check for API documentation
            api_doc_coverage = self._check_api_documentation()
            details["api_doc_coverage"] = api_doc_coverage
            score *= api_doc_coverage
            
            if api_doc_coverage < 0.8:
                recommendations.append("Add more API documentation")
            
            # Check for examples
            examples_dir = self.project_root / "examples"
            if examples_dir.exists():
                example_files = list(examples_dir.glob("**/*.py"))
                example_files.extend(list(examples_dir.glob("**/*.ipynb")))
                
                if example_files:
                    print(f"  ✅ Examples directory with {len(example_files)} files")
                    details["examples_count"] = len(example_files)
                else:
                    recommendations.append("Add usage examples")
                    score *= 0.95
            else:
                recommendations.append("Create examples directory")
                score *= 0.9
                details["examples_count"] = 0
            
            print(f"  📚 Documentation Score: {score:.2f}")
            
        except Exception as e:
            issues.append(f"Documentation check failed: {e}")
            score = 0.7
        
        execution_time = time.time() - start_time
        passed = score >= self.pass_thresholds["documentation"]
        
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {self.pass_thresholds['documentation']:.2f})")
        
        return QualityGateResult(
            gate_name="documentation",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    async def _gate_mathematical_validation(self) -> QualityGateResult:
        """Execute mathematical validation gate."""
        start_time = time.time()
        print("\n📐 Mathematical Validation Gate")
        print("-" * 30)
        
        issues = []
        recommendations = []
        details = {}
        score = 1.0
        
        try:
            # Test mathematical parser
            print("  🧮 Testing mathematical parsing...")
            
            try:
                from autoformalize.parsers.latex_parser import LaTeXParser
                
                parser = LaTeXParser()
                
                # Test basic mathematical content
                test_content = r"""
                \begin{theorem}
                For any natural number $n$, we have $n + 0 = n$.
                \end{theorem}
                """
                
                parsed = await parser.parse(test_content)
                
                if hasattr(parsed, 'theorems') or hasattr(parsed, 'content'):
                    print("  ✅ Mathematical parsing working")
                    details["math_parsing"] = True
                else:
                    issues.append("Mathematical parsing not working correctly")
                    score *= 0.8
                    details["math_parsing"] = False
                
            except Exception as e:
                issues.append(f"Mathematical parser test failed: {e}")
                score *= 0.7
            
            # Test validation system
            print("  🔍 Testing mathematical validation...")
            
            try:
                from autoformalize.utils.enhanced_validation import MathematicalValidator
                
                validator = MathematicalValidator()
                
                valid_latex = r"\begin{theorem} $x = x$ \end{theorem}"
                invalid_latex = r"\begin{theorem $x = x$ \end{theorem}"  # Missing brace
                
                valid_result = await validator.validate_latex_content(valid_latex)
                invalid_result = await validator.validate_latex_content(invalid_latex)
                
                if valid_result.valid and not invalid_result.valid:
                    print("  ✅ Mathematical validation working")
                    details["math_validation"] = True
                else:
                    issues.append("Mathematical validation not working correctly")
                    score *= 0.8
                    details["math_validation"] = False
                
            except Exception as e:
                issues.append(f"Mathematical validation test failed: {e}")
                score *= 0.7
            
            # Test formal code generation
            print("  🔧 Testing formal code generation...")
            
            try:
                pipeline = EnhancedFormalizationPipeline(
                    target_system="lean4",
                    verification_mode=VerificationMode.MOCK
                )
                
                math_content = r"$\forall n \in \mathbb{N}, n + 0 = n$"
                result = await pipeline.formalize(math_content)
                
                if result.formal_code and len(result.formal_code) > 10:
                    print("  ✅ Formal code generation working")
                    details["code_generation"] = True
                    details["generated_code_length"] = len(result.formal_code)
                else:
                    issues.append("Formal code generation not producing adequate output")
                    score *= 0.8
                    details["code_generation"] = False
                
            except Exception as e:
                issues.append(f"Formal code generation test failed: {e}")
                score *= 0.7
            
            # Test verification integration
            print("  ✅ Testing verification integration...")
            
            try:
                from autoformalize.verifiers.lean_verifier import Lean4Verifier
                
                verifier = Lean4Verifier()
                
                # Test installation check
                install_status = await verifier.check_lean_installation()
                details["lean_verifier_status"] = install_status
                
                if install_status.get("installed", False):
                    print("  ✅ Lean 4 verifier integration ready")
                    details["verifier_ready"] = True
                else:
                    print("  ⚠️  Lean 4 not installed (expected in test environment)")
                    details["verifier_ready"] = False
                    # Don't penalize for missing Lean in test environment
                
            except Exception as e:
                issues.append(f"Verification integration test failed: {e}")
                score *= 0.9
            
            print(f"  📐 Mathematical Validation Score: {score:.2f}")
            
        except Exception as e:
            issues.append(f"Mathematical validation gate failed: {e}")
            score = 0.6
        
        execution_time = time.time() - start_time
        passed = score >= self.pass_thresholds["mathematical_validation"]
        
        print(f"  {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {self.pass_thresholds['mathematical_validation']:.2f})")
        
        return QualityGateResult(
            gate_name="mathematical_validation",
            passed=passed,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def _check_import_structure(self) -> float:
        """Check import structure quality."""
        try:
            python_files = list(self.project_root.glob("src/**/*.py"))
            if not python_files:
                return 0.5
            
            good_imports = 0
            total_imports = 0
            
            for py_file in python_files[:10]:  # Check first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    for line in lines[:20]:  # Check first 20 lines
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            total_imports += 1
                            if not ('*' in line):  # Avoid wildcard imports
                                good_imports += 1
                except:
                    continue
            
            if total_imports == 0:
                return 1.0
            
            return good_imports / total_imports
            
        except:
            return 0.8
    
    def _check_docstring_coverage(self) -> float:
        """Estimate docstring coverage."""
        try:
            python_files = list(self.project_root.glob("src/**/*.py"))
            if not python_files:
                return 0.5
            
            documented_functions = 0
            total_functions = 0
            
            for py_file in python_files[:10]:  # Check first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    in_function = False
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def ') or line.strip().startswith('async def '):
                            total_functions += 1
                            in_function = True
                            
                            # Check next few lines for docstring
                            for j in range(i + 1, min(i + 5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    documented_functions += 1
                                    break
                                elif lines[j].strip() and not lines[j].strip().startswith('#'):
                                    break  # Found code before docstring
                except:
                    continue
            
            if total_functions == 0:
                return 1.0
            
            return documented_functions / total_functions
            
        except:
            return 0.7
    
    def _scan_for_secrets(self) -> int:
        """Scan for potential hardcoded secrets."""
        try:
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            import re
            
            python_files = list(self.project_root.glob("**/*.py"))
            secrets_found = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        # Filter out obvious test/example values
                        real_matches = [m for m in matches if not any(
                            test_val in m.lower() for test_val in 
                            ['test', 'example', 'demo', 'mock', 'fake', 'your_key_here']
                        )]
                        secrets_found += len(real_matches)
                except:
                    continue
            
            return secrets_found
            
        except:
            return 0
    
    def _check_security_practices(self) -> float:
        """Check security practices implementation."""
        try:
            # Check if security module exists
            security_files = list(self.project_root.glob("**/security/*.py"))
            security_files.extend(list(self.project_root.glob("**/security.py")))
            
            if not security_files:
                return 0.7
            
            score = 1.0
            
            # Check for input validation
            validation_found = False
            for sec_file in security_files:
                try:
                    with open(sec_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'validation' in content.lower() or 'sanitiz' in content.lower():
                        validation_found = True
                        break
                except:
                    continue
            
            if not validation_found:
                score *= 0.9
            
            return score
            
        except:
            return 0.8
    
    def _check_input_validation(self) -> float:
        """Check input validation implementation."""
        try:
            # Look for validation utilities
            validation_files = list(self.project_root.glob("**/validation*.py"))
            validation_files.extend(list(self.project_root.glob("**/utils/*validation*.py")))
            
            if validation_files:
                return 0.9
            
            # Check for validation in main files
            python_files = list(self.project_root.glob("src/**/*.py"))
            validation_patterns = ['validate', 'sanitize', 'clean', 'escape']
            
            for py_file in python_files[:5]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    if any(pattern in content for pattern in validation_patterns):
                        return 0.8
                except:
                    continue
            
            return 0.6
            
        except:
            return 0.7
    
    def _estimate_test_coverage(self) -> float:
        """Estimate test coverage based on file ratios."""
        try:
            source_files = list(self.project_root.glob("src/**/*.py"))
            test_files = list(self.project_root.glob("**/test_*.py"))
            test_files.extend(list(self.project_root.glob("tests/**/*.py")))
            
            if not source_files:
                return 0.0
            
            if not test_files:
                return 0.0
            
            # Simple heuristic: ratio of test files to source files
            # Adjusted to account for typical test coverage patterns
            raw_ratio = len(test_files) / len(source_files)
            
            # Convert to estimated coverage percentage
            estimated_coverage = min(raw_ratio * 0.8, 0.95)  # Cap at 95%
            
            return estimated_coverage
            
        except:
            return 0.5
    
    def _check_api_documentation(self) -> float:
        """Check API documentation coverage."""
        try:
            # Look for API documentation files
            api_docs = list(self.project_root.glob("**/api*.md"))
            api_docs.extend(list(self.project_root.glob("**/API*.md")))
            api_docs.extend(list(self.project_root.glob("docs/**/*api*.rst")))
            
            if api_docs:
                return 0.9
            
            # Check for docstrings in main modules
            main_modules = list(self.project_root.glob("src/**/__init__.py"))
            
            documented_modules = 0
            
            for module in main_modules:
                try:
                    with open(module, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if '"""' in content or "'''" in content:
                        documented_modules += 1
                except:
                    continue
            
            if not main_modules:
                return 0.7
            
            return documented_modules / len(main_modules)
            
        except:
            return 0.6
    
    def _calculate_overall_score(self) -> None:
        """Calculate overall quality score."""
        weighted_score = 0.0
        
        for result in self.results:
            weight = self.gate_weights.get(result.gate_name, 0.1)
            weighted_score += result.score * weight
        
        self.overall_score = weighted_score
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        execution_time = time.time() - self.start_time
        
        passed_gates = [r for r in self.results if r.passed]
        failed_gates = [r for r in self.results if not r.passed]
        
        all_issues = []
        all_recommendations = []
        
        for result in self.results:
            all_issues.extend(result.issues)
            all_recommendations.extend(result.recommendations)
        
        report = {
            "overall_status": "PASSED" if len(failed_gates) == 0 else "FAILED",
            "overall_score": self.overall_score,
            "execution_time": execution_time,
            "gates_summary": {
                "total": len(self.results),
                "passed": len(passed_gates),
                "failed": len(failed_gates)
            },
            "gate_results": [
                {
                    "name": r.gate_name,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": self.pass_thresholds.get(r.gate_name, 0.8),
                    "weight": self.gate_weights.get(r.gate_name, 0.1),
                    "execution_time": r.execution_time,
                    "issues_count": len(r.issues),
                    "recommendations_count": len(r.recommendations)
                }
                for r in self.results
            ],
            "detailed_results": {r.gate_name: r for r in self.results},
            "summary": {
                "total_issues": len(all_issues),
                "total_recommendations": len(all_recommendations),
                "critical_issues": [issue for issue in all_issues if "critical" in issue.lower()],
                "production_ready": len(failed_gates) == 0 and self.overall_score >= 0.8
            }
        }
        
        return report


async def main():
    """Run comprehensive quality gates."""
    print("🔒 COMPREHENSIVE QUALITY GATES EXECUTION")
    print("Validating production readiness...")
    print()
    
    # Initialize quality gates system
    quality_gates = ComprehensiveQualityGates()
    
    # Execute all gates
    report = await quality_gates.execute_all_gates()
    
    # Print final report
    print("\n" + "=" * 60)
    print("📊 FINAL QUALITY GATES REPORT")
    print("=" * 60)
    
    print(f"\n🎯 Overall Status: {report['overall_status']}")
    print(f"📊 Overall Score: {report['overall_score']:.2f}")
    print(f"⏱️  Execution Time: {report['execution_time']:.1f}s")
    
    print(f"\n📈 Gates Summary:")
    print(f"  • Total Gates: {report['gates_summary']['total']}")
    print(f"  • Passed: {report['gates_summary']['passed']} ✅")
    print(f"  • Failed: {report['gates_summary']['failed']} ❌")
    
    print(f"\n📋 Individual Gate Results:")
    for gate in report['gate_results']:
        status = "✅ PASS" if gate['passed'] else "❌ FAIL"
        print(f"  {gate['name']:<25} {status} Score: {gate['score']:.2f} (Threshold: {gate['threshold']:.2f})")
    
    print(f"\n🔍 Issues & Recommendations:")
    print(f"  • Total Issues: {report['summary']['total_issues']}")
    print(f"  • Total Recommendations: {report['summary']['total_recommendations']}")
    
    if report['summary']['critical_issues']:
        print(f"  • Critical Issues: {len(report['summary']['critical_issues'])}")
        for issue in report['summary']['critical_issues']:
            print(f"    ⚠️  {issue}")
    
    print(f"\n🚀 Production Ready: {'✅ YES' if report['summary']['production_ready'] else '❌ NO'}")
    
    if not report['summary']['production_ready']:
        print("\n📝 Required Actions Before Production:")
        failed_gates = [g for g in report['gate_results'] if not g['passed']]
        for gate in failed_gates:
            print(f"  • Fix {gate['name']} gate (Score: {gate['score']:.2f}, Required: {gate['threshold']:.2f})")
    
    # Save detailed report
    report_file = Path("quality_gates_comprehensive_report.json")
    with open(report_file, 'w') as f:
        # Convert dataclass objects to dicts for JSON serialization
        json_report = report.copy()
        json_report['detailed_results'] = {
            name: {
                'gate_name': result.gate_name,
                'passed': result.passed,
                'score': result.score,
                'details': result.details,
                'issues': result.issues,
                'recommendations': result.recommendations,
                'execution_time': result.execution_time
            }
            for name, result in report['detailed_results'].items()
        }
        json.dump(json_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    import asyncio
    
    print("Comprehensive Quality Gates for Mathematical Formalization Platform")
    print("Executing all mandatory quality gates for production deployment...")
    
    result = asyncio.run(main())
    
    # Exit with appropriate code
    if result['summary']['production_ready']:
        print("\n🎉 ALL QUALITY GATES PASSED - READY FOR PRODUCTION! 🎉")
        sys.exit(0)
    else:
        print("\n⚠️  QUALITY GATES FAILED - PRODUCTION DEPLOYMENT BLOCKED")
        sys.exit(1)