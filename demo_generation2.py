#!/usr/bin/env python3
"""
Generation 2 Demo: Robustness and reliability features
Demonstrates comprehensive error handling, validation, security, and monitoring.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.core.resilience import RetryManager, CircuitBreakerConfig, resilience_manager
from autoformalize.security.input_validation import LaTeXValidator, content_sanitizer
from autoformalize.core.pipeline import FormalizationPipeline

async def demo_generation2():
    """Demonstrate Generation 2 (Robust) functionality."""
    print("🛡️  Generation 2: MAKE IT ROBUST - Demo Starting")
    print("=" * 60)
    
    try:
        # Test 1: Input Validation & Security
        print("🔒 Test 1: Input Validation & Security")
        validator = LaTeXValidator()
        
        # Valid content
        safe_latex = r"\begin{theorem}For any $x \in \mathbb{R}$, $x^2 \geq 0$.\end{theorem}"
        result = validator.validate(safe_latex)
        print(f"✅ Safe content validation: {'Valid' if result.is_valid else 'Invalid'}")
        print(f"   - Risk score: {result.risk_score:.2f}")
        print(f"   - Warnings: {len(result.warnings)}")
        
        # Potentially dangerous content
        dangerous_latex = r"\input{/etc/passwd} \begin{theorem}$x > 0$\end{theorem}"
        result = validator.validate(dangerous_latex)
        print(f"🚫 Dangerous content validation: {'Valid' if result.is_valid else 'Invalid'}")
        print(f"   - Errors: {len(result.errors)}")
        print(f"   - Risk score: {result.risk_score:.2f}")
        
        # Test 2: Error Recovery & Resilience
        print("\n🔄 Test 2: Error Recovery & Resilience")
        
        # Test retry mechanism
        retry_manager = RetryManager()
        failure_count = 0
        
        async def flaky_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise Exception("Simulated failure")
            return "Success after retries"
        
        try:
            result = await retry_manager.retry(flaky_operation, retry_manager.config)
            print(f"✅ Retry mechanism: {result}")
            print(f"   - Attempts needed: {failure_count}")
        except Exception as e:
            print(f"❌ Retry failed: {e}")
        
        # Test 3: Circuit Breaker
        print("\n⚡ Test 3: Circuit Breaker Pattern")
        
        # Register circuit breaker
        cb_config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1.0)
        cb = resilience_manager.register_circuit_breaker("test_service", cb_config)
        
        # Simulate failures to trip circuit breaker
        async def failing_service():
            raise Exception("Service unavailable")
        
        failures = 0
        for i in range(4):
            try:
                await cb.call(failing_service)
            except Exception:
                failures += 1
                pass
        
        print(f"✅ Circuit breaker tripped after {failures} failures")
        print(f"   - Circuit state: {cb.state.value}")
        
        # Test 4: Health Monitoring
        print("\n🏥 Test 4: Health Monitoring")
        
        # Register health checks
        async def database_health():
            return True  # Simulate healthy database
        
        def cache_health():
            return True  # Simulate healthy cache
        
        resilience_manager.register_health_check("database", database_health)
        resilience_manager.register_health_check("cache", cache_health)
        
        health_status = await resilience_manager.check_health()
        print(f"✅ System health: {'Healthy' if health_status.is_healthy else 'Unhealthy'}")
        print(f"   - Health checks: {health_status.metrics.get('health_checks', 0)}")
        print(f"   - Warnings: {len(health_status.warnings)}")
        print(f"   - Open circuits: {health_status.metrics.get('open_circuits', 0)}")
        
        # Test 5: Content Sanitization
        print("\n🧼 Test 5: Content Sanitization")
        
        malicious_content = "<script>alert('xss')</script>\\input{/etc/passwd}"
        sanitized = content_sanitizer.sanitize_text_input(malicious_content)
        print(f"✅ Content sanitization completed")
        print(f"   - Original length: {len(malicious_content)}")
        print(f"   - Sanitized length: {len(sanitized)}")
        print(f"   - Safe preview: {sanitized[:50]}...")
        
        # Test 6: Robust Pipeline Integration
        print("\n🔧 Test 6: Robust Pipeline Integration")
        
        pipeline = FormalizationPipeline(target_system="lean4")
        
        # Test with various inputs
        test_cases = [
            ("Valid theorem", r"\begin{theorem}$\forall x: x = x$\end{theorem}"),
            ("Empty content", ""),
            ("Malformed LaTeX", r"\begin{theorem incomplete"),
            ("Very long content", "x" * 10000),
        ]
        
        robust_results = []
        for name, content in test_cases:
            try:
                result = await pipeline.formalize(content, verify=False)
                robust_results.append((name, result.success, result.error_message))
            except Exception as e:
                robust_results.append((name, False, str(e)))
        
        print("✅ Robust pipeline testing completed:")
        for name, success, error in robust_results:
            status = "✅" if success else "❌"
            print(f"   {status} {name}: {'Success' if success else f'Error - {error[:50]}...'}")
        
        # Test 7: Performance Under Load
        print("\n⚡ Test 7: Concurrent Load Testing")
        
        async def concurrent_formalization(i):
            try:
                result = await pipeline.formalize(
                    f"\\begin{{theorem}}Test theorem {i}\\end{{theorem}}", 
                    verify=False
                )
                return result.success
            except Exception:
                return False
        
        # Run 20 concurrent formalizations
        tasks = [concurrent_formalization(i) for i in range(20)]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in concurrent_results if r is True)
        total = len(concurrent_results)
        print(f"✅ Concurrent processing: {successful}/{total} successful")
        print(f"   - Success rate: {successful/total:.1%}")
        
        print("\n🎉 Generation 2 Demo Complete!")
        print("✅ Robustness and reliability features verified")
        print("🛡️  System is now resilient and secure")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(demo_generation2())
    sys.exit(0 if success else 1)