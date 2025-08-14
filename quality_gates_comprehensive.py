#!/usr/bin/env python3
"""
Comprehensive Quality Gates Implementation
Validates code quality, security, performance, and functionality.
"""

import asyncio
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.core.pipeline import FormalizationPipeline
from autoformalize.security.input_validation import LaTeXValidator
from autoformalize.performance.adaptive_optimizer import get_performance_optimizer

class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str):
        self.name = name
        
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check quality gate. Returns (passed, message, details)."""
        raise NotImplementedError

class CodeQualityGate(QualityGate):
    """Code quality and static analysis gate."""
    
    def __init__(self):
        super().__init__("Code Quality")
    
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check code quality metrics."""
        details = {
            'python_files_count': 0,
            'total_lines': 0,
            'complexity_issues': 0,
            'style_issues': 0
        }
        
        try:
            # Count Python files and lines
            src_path = Path(__file__).parent / "src"
            py_files = list(src_path.rglob("*.py"))
            details['python_files_count'] = len(py_files)
            
            total_lines = 0
            for py_file in py_files:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
            
            details['total_lines'] = total_lines
            
            # Simple complexity check (functions per file)
            avg_file_size = total_lines / len(py_files) if py_files else 0
            if avg_file_size > 500:
                details['complexity_issues'] = 1
            
            # Check for basic style issues
            style_issues = 0
            for py_file in py_files:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '\t' in content:  # Tabs instead of spaces
                        style_issues += 1
            
            details['style_issues'] = style_issues
            
            passed = (
                details['complexity_issues'] == 0 and
                details['style_issues'] < 5 and  # Allow some style issues
                details['python_files_count'] > 0
            )
            
            message = "Code quality check passed" if passed else "Code quality issues detected"
            return passed, message, details
            
        except Exception as e:
            return False, f"Code quality check failed: {e}", details

class SecurityGate(QualityGate):
    """Security vulnerability and validation gate."""
    
    def __init__(self):
        super().__init__("Security")
    
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check security vulnerabilities."""
        details = {
            'validation_test_passed': False,
            'dangerous_patterns': 0,
            'input_sanitization': False
        }
        
        try:
            # Test input validation
            validator = LaTeXValidator()
            
            # Test with safe content
            safe_result = validator.validate(r"\begin{theorem}Safe theorem\end{theorem}")
            
            # Test with dangerous content
            dangerous_content = r"\input{/etc/passwd} \begin{theorem}Test\end{theorem}"
            dangerous_result = validator.validate(dangerous_content)
            
            details['validation_test_passed'] = safe_result.is_valid and not dangerous_result.is_valid
            details['dangerous_patterns'] = len(dangerous_result.errors)
            details['input_sanitization'] = len(dangerous_result.sanitized_content or "") > 0
            
            passed = (
                details['validation_test_passed'] and
                details['dangerous_patterns'] > 0 and  # Should detect dangerous patterns
                details['input_sanitization']
            )
            
            message = "Security checks passed" if passed else "Security vulnerabilities detected"
            return passed, message, details
            
        except Exception as e:
            return False, f"Security check failed: {e}", details

class PerformanceGate(QualityGate):
    """Performance and scalability gate."""
    
    def __init__(self):
        super().__init__("Performance")
    
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check performance metrics."""
        details = {
            'latency_ms': 0,
            'throughput_ops_sec': 0,
            'memory_usage_mb': 0,
            'cache_hit_rate': 0.0,
            'concurrent_processing': False
        }
        
        try:
            # Test basic latency
            start_time = time.time()
            pipeline = FormalizationPipeline(target_system="lean4")
            theorem = r"\begin{theorem}Performance test: $x = x$\end{theorem}"
            result = await pipeline.formalize(theorem, verify=False)
            latency = (time.time() - start_time) * 1000
            
            details['latency_ms'] = round(latency, 2)
            
            # Test throughput
            start_time = time.time()
            tasks = []
            for i in range(10):
                tasks.append(pipeline.formalize(f"\\begin{{theorem}}Test {i}\\end{{theorem}}", verify=False))
            
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            throughput = len(results) / duration if duration > 0 else 0
            
            details['throughput_ops_sec'] = round(throughput, 1)
            details['concurrent_processing'] = all(r.success for r in results)
            
            # Check performance optimizer
            perf_optimizer = get_performance_optimizer()
            perf_stats = perf_optimizer.get_comprehensive_stats()
            
            details['cache_hit_rate'] = perf_stats['metrics']['cache_hit_rate']
            details['memory_usage_mb'] = round(perf_stats['metrics']['memory_usage_mb'], 1)
            
            passed = (
                details['latency_ms'] < 1000 and  # Less than 1 second
                details['throughput_ops_sec'] > 10 and  # At least 10 ops/sec
                details['concurrent_processing'] and
                details['memory_usage_mb'] < 1000  # Less than 1GB
            )
            
            message = "Performance checks passed" if passed else "Performance issues detected"
            return passed, message, details
            
        except Exception as e:
            return False, f"Performance check failed: {e}", details

class FunctionalityGate(QualityGate):
    """Core functionality and integration gate."""
    
    def __init__(self):
        super().__init__("Functionality")
    
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check core functionality."""
        details = {
            'latex_parsing': False,
            'pipeline_integration': False,
            'multi_system_support': False,
            'error_handling': False,
            'concurrent_processing': False
        }
        
        try:
            # Test LaTeX parsing
            from autoformalize.parsers.latex_parser import LaTeXParser
            parser = LaTeXParser()
            theorem = r"\begin{theorem}Functional test theorem\end{theorem}"
            parsed = await parser.parse(theorem)
            details['latex_parsing'] = len(parsed.theorems) > 0
            
            # Test pipeline integration
            pipeline = FormalizationPipeline(target_system="lean4")
            result = await pipeline.formalize(theorem, verify=False)
            details['pipeline_integration'] = result.success
            
            # Test multi-system support
            systems_tested = []
            for system in ["lean4", "isabelle", "coq"]:
                try:
                    pipeline = FormalizationPipeline(target_system=system)
                    result = await pipeline.formalize(theorem, verify=False)
                    if result.success:
                        systems_tested.append(system)
                except Exception:
                    pass
            
            details['multi_system_support'] = len(systems_tested) >= 2
            
            # Test error handling
            pipeline = FormalizationPipeline(target_system="lean4")
            error_result = await pipeline.formalize("", verify=False)  # Empty input
            details['error_handling'] = not error_result.success and error_result.error_message
            
            # Test concurrent processing
            tasks = []
            for i in range(5):
                tasks.append(pipeline.formalize(f"\\begin{{theorem}}Concurrent {i}\\end{{theorem}}", verify=False))
            
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in concurrent_results if not isinstance(r, Exception) and r.success)
            details['concurrent_processing'] = successful >= 4  # At least 80% success
            
            passed = all(details.values())
            message = "Functionality checks passed" if passed else "Functionality issues detected"
            return passed, message, details
            
        except Exception as e:
            return False, f"Functionality check failed: {e}", details

class ReliabilityGate(QualityGate):
    """System reliability and resilience gate."""
    
    def __init__(self):
        super().__init__("Reliability")
    
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check system reliability."""
        details = {
            'retry_mechanism': False,
            'circuit_breaker': False,
            'health_monitoring': False,
            'error_recovery': False,
            'graceful_degradation': False
        }
        
        try:
            # Test retry mechanism
            from autoformalize.core.resilience import RetryManager, RetryConfig
            retry_manager = RetryManager(RetryConfig(max_attempts=2, base_delay=0.1))
            
            attempt_count = 0
            async def flaky_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 2:
                    raise Exception("Simulated failure")
                return "Success"
            
            result = await retry_manager.retry(flaky_function, retry_manager.config)
            details['retry_mechanism'] = result == "Success" and attempt_count == 2
            
            # Test circuit breaker
            from autoformalize.core.resilience import CircuitBreakerConfig, resilience_manager
            cb_config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
            cb = resilience_manager.register_circuit_breaker("test_reliability", cb_config)
            
            # Trigger circuit breaker
            try:
                await cb.call(lambda: exec('raise Exception("Test failure")'))
            except:
                pass
            
            details['circuit_breaker'] = cb.state.value == "open"
            
            # Test health monitoring
            resilience_manager.register_health_check("test_health", lambda: True)
            health_status = await resilience_manager.check_health()
            details['health_monitoring'] = health_status.is_healthy
            
            # Test error recovery
            pipeline = FormalizationPipeline(target_system="lean4")
            malformed_input = "\\begin{theorem incomplete"
            error_result = await pipeline.formalize(malformed_input, verify=False)
            details['error_recovery'] = not error_result.success and error_result.error_message is not None
            
            # Test graceful degradation (pipeline works even with missing components)
            pipeline = FormalizationPipeline(target_system="lean4")
            result = await pipeline.formalize(r"\begin{theorem}Test\end{theorem}", verify=False)
            details['graceful_degradation'] = result.success  # Should work even without real LLM
            
            passed = sum(details.values()) >= 4  # At least 4/5 checks pass
            message = "Reliability checks passed" if passed else "Reliability issues detected"
            return passed, message, details
            
        except Exception as e:
            return False, f"Reliability check failed: {e}", details

class DeploymentReadinessGate(QualityGate):
    """Production deployment readiness gate."""
    
    def __init__(self):
        super().__init__("Deployment Readiness")
    
    async def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check deployment readiness."""
        details = {
            'dockerfile_exists': False,
            'requirements_complete': False,
            'environment_config': False,
            'monitoring_ready': False,
            'documentation_available': False
        }
        
        try:
            repo_root = Path(__file__).parent
            
            # Check Dockerfile
            dockerfile = repo_root / "Dockerfile"
            details['dockerfile_exists'] = dockerfile.exists()
            
            # Check requirements
            requirements = repo_root / "requirements.txt"
            pyproject = repo_root / "pyproject.toml"
            details['requirements_complete'] = requirements.exists() or pyproject.exists()
            
            # Check environment configuration
            env_files = [
                repo_root / ".env.example",
                repo_root / "docker-compose.yml",
                repo_root / "deployment"
            ]
            details['environment_config'] = any(f.exists() for f in env_files)
            
            # Check monitoring setup
            monitoring_files = [
                repo_root / "docs" / "monitoring",
                repo_root / "docker" / "prometheus.yml"
            ]
            details['monitoring_ready'] = any(f.exists() for f in monitoring_files)
            
            # Check documentation
            docs = [
                repo_root / "README.md",
                repo_root / "docs",
                repo_root / "DEPLOYMENT.md"
            ]
            details['documentation_available'] = any(f.exists() for f in docs)
            
            passed = sum(details.values()) >= 4  # At least 4/5 checks pass
            message = "Deployment readiness passed" if passed else "Deployment readiness issues"
            return passed, message, details
            
        except Exception as e:
            return False, f"Deployment readiness check failed: {e}", details

async def run_quality_gates() -> Dict[str, Any]:
    """Run all quality gates and return comprehensive results."""
    print("🧪 COMPREHENSIVE QUALITY GATES EXECUTION")
    print("=" * 60)
    
    gates = [
        CodeQualityGate(),
        SecurityGate(),
        PerformanceGate(),
        FunctionalityGate(),
        ReliabilityGate(),
        DeploymentReadinessGate()
    ]
    
    results = {
        'timestamp': time.time(),
        'total_gates': len(gates),
        'passed_gates': 0,
        'failed_gates': 0,
        'overall_passed': False,
        'gate_results': {},
        'summary': {}
    }
    
    for gate in gates:
        print(f"\n🔍 Running {gate.name} Gate...")
        
        try:
            start_time = time.time()
            passed, message, details = await gate.check()
            duration = time.time() - start_time
            
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status}: {message} ({duration:.2f}s)")
            
            # Print key details
            for key, value in details.items():
                if isinstance(value, bool):
                    status_icon = "✅" if value else "❌"
                    print(f"     {status_icon} {key}: {value}")
                else:
                    print(f"     📊 {key}: {value}")
            
            results['gate_results'][gate.name] = {
                'passed': passed,
                'message': message,
                'details': details,
                'duration': duration
            }
            
            if passed:
                results['passed_gates'] += 1
            else:
                results['failed_gates'] += 1
                
        except Exception as e:
            print(f"   ❌ FAILED: {gate.name} gate crashed: {e}")
            results['gate_results'][gate.name] = {
                'passed': False,
                'message': f"Gate crashed: {e}",
                'details': {},
                'duration': 0
            }
            results['failed_gates'] += 1
    
    # Calculate overall result
    results['overall_passed'] = results['passed_gates'] >= (results['total_gates'] * 0.8)  # 80% pass rate
    
    # Generate summary
    results['summary'] = {
        'pass_rate': results['passed_gates'] / results['total_gates'],
        'critical_failures': results['failed_gates'],
        'recommendation': "Deploy ready" if results['overall_passed'] else "Fix issues before deployment"
    }
    
    print(f"\n📊 QUALITY GATES SUMMARY")
    print(f"   Total Gates: {results['total_gates']}")
    print(f"   Passed: {results['passed_gates']}")
    print(f"   Failed: {results['failed_gates']}")
    print(f"   Pass Rate: {results['summary']['pass_rate']:.1%}")
    print(f"   Overall Result: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
    print(f"   Recommendation: {results['summary']['recommendation']}")
    
    return results

async def main():
    """Main execution function."""
    try:
        results = await run_quality_gates()
        
        # Save results to file
        results_path = Path(__file__).parent / "quality_gates_results.json"
        with open(results_path, 'w') as f:
            # Convert non-serializable objects
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_path}")
        
        # Return appropriate exit code
        return 0 if results['overall_passed'] else 1
        
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)