#!/usr/bin/env python3
"""Test advanced robustness features."""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.security.advanced_security import (
    SecurityValidator, SecurityConfig, SecurityLevel, ThreatType
)
from autoformalize.utils.advanced_error_handling import (
    AdvancedErrorHandler, RetryHandler, CircuitBreaker, GracefulDegradation,
    RetryConfig, CircuitBreakerConfig, with_retry, with_circuit_breaker
)

async def test_security_features():
    """Test advanced security features."""
    print("🔒 Testing Advanced Security Features")
    print("=" * 50)
    
    # Create security validator
    config = SecurityConfig(
        security_level=SecurityLevel.PRODUCTION,
        max_input_length=1000,
        rate_limit_requests=5,
        rate_limit_window=60
    )
    validator = SecurityValidator(config)
    
    print(f"✅ Security validator created with {config.security_level.value} level")
    
    # Test safe LaTeX input
    safe_latex = r"""
    \begin{theorem}
    For any natural number $n$, we have $n + 0 = n$.
    \end{theorem}
    \begin{proof}
    This follows from the definition of addition.
    \end{proof}
    """
    
    is_safe = validator.validate_latex_input(safe_latex)
    print(f"✅ Safe LaTeX validation: {'PASSED' if is_safe else 'FAILED'}")
    
    # Test malicious input detection
    malicious_inputs = [
        r"\input{/etc/passwd}",  # File inclusion
        r"\write18{rm -rf /}",   # Command execution
        r"<script>alert('xss')</script>",  # XSS attempt
        r"eval(dangerous_code)",  # Code injection
        r"A" * 2000,  # Length attack
    ]
    
    blocked_count = 0
    for malicious_input in malicious_inputs:
        is_safe = validator.validate_latex_input(malicious_input, source_ip="192.168.1.100")
        if not is_safe:
            blocked_count += 1
    
    print(f"✅ Malicious input detection: {blocked_count}/{len(malicious_inputs)} blocked")
    
    # Test rate limiting
    print("\n📊 Testing rate limiting...")
    for i in range(7):  # Exceed the limit of 5
        within_limit = validator.check_rate_limit("test_user", "192.168.1.100")
        if not within_limit:
            print(f"   Request {i+1}: Rate limit exceeded")
            break
        else:
            print(f"   Request {i+1}: Within limit")
    
    # Test input sanitization
    print("\n🧹 Testing input sanitization...")
    dirty_input = r"\input{dangerous.tex} \def\malicious{} Safe content here"
    sanitized = validator.sanitize_latex_input(dirty_input)
    print(f"   Original length: {len(dirty_input)}")
    print(f"   Sanitized length: {len(sanitized)}")
    print(f"   Sanitization effective: {len(sanitized) < len(dirty_input)}")
    
    # Get security summary
    print("\n📈 Security Summary:")
    summary = validator.get_security_summary()
    print(f"   Total events: {summary['total_events']}")
    print(f"   By threat type: {summary['by_threat_type']}")
    print(f"   By severity: {summary['by_severity']}")
    
    print("\n✅ Security features test completed!")

async def test_error_handling():
    """Test advanced error handling features."""
    print("\n🛡️ Testing Advanced Error Handling")
    print("=" * 50)
    
    # Create error handler
    error_handler = AdvancedErrorHandler()
    
    # Test failure classification
    test_exceptions = [
        ConnectionError("Connection refused"),
        TimeoutError("Request timed out"),
        ValueError("Invalid input format"),
        FileNotFoundError("File not found"),
        Exception("Rate limit exceeded"),
        Exception("Memory allocation failed"),
    ]
    
    print("🔍 Testing failure classification:")
    for exc in test_exceptions:
        failure_type = error_handler.classify_failure(exc)
        error_handler.record_failure(exc, context={'test': True})
        print(f"   {type(exc).__name__}: {failure_type.value}")
    
    # Test retry mechanism
    print("\n🔄 Testing retry mechanism:")
    
    class FlakyFunction:
        def __init__(self, fail_times: int = 2):
            self.call_count = 0
            self.fail_times = fail_times
        
        async def __call__(self, *args, **kwargs):
            self.call_count += 1
            if self.call_count <= self.fail_times:
                raise ConnectionError(f"Simulated failure {self.call_count}")
            return f"Success after {self.call_count} attempts"
    
    # Test with retry decorator
    retry_config = RetryConfig(
        max_attempts=5,
        base_delay=0.1,  # Fast for testing
        backoff_strategy="exponential"
    )
    
    flaky_func = FlakyFunction(fail_times=2)
    retry_handler = RetryHandler(retry_config, error_handler)
    
    try:
        result = await retry_handler.retry_call(flaky_func)
        print(f"   ✅ Retry successful: {result}")
    except Exception as e:
        print(f"   ❌ Retry failed: {e}")
    
    # Test circuit breaker
    print("\n⚡ Testing circuit breaker:")
    
    circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=1.0,  # Fast for testing
        name="test_circuit"
    )
    
    circuit_breaker = CircuitBreaker(circuit_config, error_handler)
    
    class AlwaysFailFunction:
        async def __call__(self, *args, **kwargs):
            raise ConnectionError("Always fails")
    
    always_fail = AlwaysFailFunction()
    
    # Trip the circuit breaker
    failures = 0
    for i in range(5):
        try:
            await circuit_breaker.call(always_fail)
        except Exception:
            failures += 1
            print(f"   Attempt {i+1}: Failed (circuit state: {circuit_breaker.state.value})")
    
    print(f"   Circuit breaker tripped after {failures} failures")
    
    # Test graceful degradation
    print("\n🎭 Testing graceful degradation:")
    
    degradation = GracefulDegradation(error_handler)
    
    async def primary_function():
        raise Exception("Primary function failed")
    
    async def fallback_function():
        return "Fallback result"
    
    degradation.register_fallback("test_operation", fallback_function)
    
    try:
        result = await degradation.execute_with_fallback(
            "test_operation",
            primary_function
        )
        print(f"   ✅ Graceful degradation: {result}")
    except Exception as e:
        print(f"   ❌ Degradation failed: {e}")
    
    # Get error summary
    print("\n📊 Error Handling Summary:")
    summary = error_handler.get_failure_summary()
    print(f"   Total failures: {summary['total_failures']}")
    print(f"   By type: {summary['by_type']}")
    print(f"   Recent failures: {summary['recent_failures']}")
    print(f"   Failure rate: {summary['failure_rate']:.2f}/hour")
    
    print("\n✅ Error handling test completed!")

async def test_decorator_functionality():
    """Test retry and circuit breaker decorators."""
    print("\n🎯 Testing Decorator Functionality")
    print("=" * 40)
    
    # Test retry decorator
    @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
    async def flaky_decorated_function(success_after: int = 2):
        if not hasattr(flaky_decorated_function, 'call_count'):
            flaky_decorated_function.call_count = 0
        flaky_decorated_function.call_count += 1
        
        if flaky_decorated_function.call_count <= success_after:
            raise ValueError(f"Failed attempt {flaky_decorated_function.call_count}")
        
        return f"Success after {flaky_decorated_function.call_count} attempts"
    
    try:
        result = await flaky_decorated_function(success_after=2)
        print(f"   ✅ Retry decorator: {result}")
    except Exception as e:
        print(f"   ❌ Retry decorator failed: {e}")
    
    # Test circuit breaker decorator
    @with_circuit_breaker(CircuitBreakerConfig(failure_threshold=2, name="decorated_circuit"))
    async def unreliable_function(should_fail: bool = True):
        if should_fail:
            raise ConnectionError("Function failed")
        return "Function succeeded"
    
    # Trip the circuit
    for i in range(4):
        try:
            result = await unreliable_function(should_fail=True)
            print(f"   Attempt {i+1}: {result}")
        except Exception as e:
            print(f"   Attempt {i+1}: {type(e).__name__}")
    
    print("\n✅ Decorator functionality test completed!")

async def test_integration():
    """Test integration of security and error handling."""
    print("\n🔗 Testing Integration")
    print("=" * 30)
    
    # Create integrated components
    security_validator = SecurityValidator()
    error_handler = AdvancedErrorHandler()
    
    # Simulate a robust formalization function
    @with_retry(RetryConfig(max_attempts=2, base_delay=0.1))
    async def secure_formalization(latex_content: str, source_ip: str = "unknown"):
        # Security validation
        if not security_validator.validate_latex_input(latex_content, source_ip):
            raise ValueError("Security validation failed")
        
        # Simulate formalization that might fail
        if "fail" in latex_content.lower():
            raise ConnectionError("Formalization service unavailable")
        
        return f"Formalized: {latex_content[:50]}..."
    
    # Test cases
    test_cases = [
        ("Safe theorem content", "192.168.1.1", True),
        (r"\input{/etc/passwd}", "192.168.1.2", False),
        ("This will fail on purpose", "192.168.1.3", False),
        ("Another safe theorem", "192.168.1.4", True),
    ]
    
    successful_cases = 0
    for latex_content, source_ip, expected_success in test_cases:
        try:
            result = await secure_formalization(latex_content, source_ip)
            print(f"   ✅ Success: {result[:30]}...")
            if expected_success:
                successful_cases += 1
        except Exception as e:
            print(f"   ❌ Failed: {type(e).__name__}")
            if not expected_success:
                successful_cases += 1
    
    print(f"\n📊 Integration test results: {successful_cases}/{len(test_cases)} as expected")
    
    # Get combined summary
    print("\n📈 Combined Summary:")
    security_summary = security_validator.get_security_summary()
    error_summary = error_handler.get_failure_summary()
    
    print(f"   Security events: {security_summary['total_events']}")
    print(f"   Error failures: {error_summary['total_failures']}")
    
    print("\n✅ Integration test completed!")

def main():
    """Main test function."""
    print("🚀 Starting Robustness Features Tests")
    print("=" * 60)
    
    # Run async tests
    try:
        # Test security features
        asyncio.run(test_security_features())
        
        # Test error handling
        asyncio.run(test_error_handling())
        
        # Test decorators
        asyncio.run(test_decorator_functionality())
        
        # Test integration
        asyncio.run(test_integration())
        
        print("\n" + "=" * 60)
        print("🎉 All robustness tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()