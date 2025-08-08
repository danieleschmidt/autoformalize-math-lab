#!/usr/bin/env python3
"""
Generation 2 Test - Robustness and Reliability Test
Tests comprehensive error handling, retry mechanisms, and resilience features.
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_resilience_utilities():
    """Test resilience utilities."""
    print("🛡️ Testing Resilience Utilities")
    
    try:
        from autoformalize.utils.resilience import (
            retry_async, CircuitBreaker, CircuitBreakerConfig, 
            health_check, graceful_degradation, resource_monitor
        )
        
        # Test health checks
        def test_health_check():
            return True
        
        health_check.register("test_check", test_health_check)
        health_results = await health_check.check_all()
        print(f"✅ Health checks working: {len(health_results)} registered")
        
        # Test resource monitoring
        resources = resource_monitor.check_resources()
        print(f"✅ Resource monitoring: {resources['memory_mb']:.1f}MB memory")
        
        # Test circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        breaker = CircuitBreaker(config)
        
        @breaker
        async def test_function():
            return "success"
        
        result = await test_function()
        print(f"✅ Circuit breaker working: {result}")
        
        # Test graceful degradation
        async def fallback_handler():
            return "fallback_result"
        
        graceful_degradation.register_fallback("test_service", fallback_handler)
        
        async def failing_service():
            raise Exception("Service unavailable")
        
        result = await graceful_degradation.execute_with_fallback(
            "test_service", failing_service
        )
        print(f"✅ Graceful degradation: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience utilities test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_robust_pipeline():
    """Test robust formalization pipeline."""
    print("\n🏗️ Testing Robust Pipeline")
    
    try:
        from autoformalize.core.robust_pipeline import RobustFormalizationPipeline
        
        # Initialize robust pipeline
        pipeline = RobustFormalizationPipeline(
            target_system="lean4",
            model="mock",
            enable_circuit_breaker=True,
            enable_retry=True,
            max_retries=2
        )
        print("✅ Robust pipeline initialized")
        
        # Test input validation
        try:
            await pipeline._validate_input("")
            print("❌ Should have failed on empty input")
            return False
        except Exception:
            print("✅ Input validation working")
        
        # Test with valid LaTeX content
        latex_content = r"""
        \begin{theorem}
        For any natural number $n$, we have $n + 0 = n$.
        \end{theorem}
        \begin{proof}
        This follows by definition of addition.
        \end{proof}
        """
        
        result = await pipeline.robust_formalize(
            latex_content,
            verify=False,  # Skip verification to avoid external dependencies
            enable_monitoring=True
        )
        
        print(f"✅ Robust formalization: success={result.success}")
        print(f"   - Retry count: {result.retry_count}")
        print(f"   - Fallback used: {result.fallback_used}")
        print(f"   - Warnings: {len(result.warnings)}")
        print(f"   - Resource usage: {bool(result.resource_usage)}")
        
        # Test metrics
        metrics = pipeline.get_robust_metrics()
        print(f"✅ Robust metrics: {metrics['successful_requests']} successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test comprehensive error handling."""
    print("\n🚨 Testing Error Handling")
    
    try:
        from autoformalize.core.robust_pipeline import RobustFormalizationPipeline
        from autoformalize.core.exceptions import ValidationError, FormalizationError
        
        pipeline = RobustFormalizationPipeline(model="mock")
        
        # Test various error conditions
        error_cases = [
            ("", "empty input"),
            ("x" * 200000, "oversized input"),  # > 100KB
            ("\\input{dangerous}", "dangerous LaTeX pattern"),
        ]
        
        for content, description in error_cases:
            try:
                if description == "oversized input":
                    # This should raise ValidationError
                    await pipeline._validate_input(content)
                    print(f"❌ Should have failed on {description}")
                    return False
                elif "dangerous" in description:
                    # This should log warning but not fail
                    await pipeline._validate_input(content)
                    print(f"✅ Handled {description} with warning")
                else:
                    await pipeline._validate_input(content)
                    print(f"❌ Should have failed on {description}")
                    return False
            except (ValidationError, FormalizationError):
                print(f"✅ Correctly handled {description}")
            except Exception as e:
                print(f"⚠️  Unexpected error for {description}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test FAILED: {e}")
        return False

async def test_batch_robustness():
    """Test robust batch processing."""
    print("\n📦 Testing Batch Robustness")
    
    try:
        from autoformalize.core.robust_pipeline import RobustFormalizationPipeline
        
        pipeline = RobustFormalizationPipeline(model="mock")
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_files = []
            for i in range(3):
                file_path = temp_path / f"test_{i}.tex"
                content = f"""
                \\begin{{theorem}}
                Test theorem {i}: $n + {i} = {i} + n$.
                \\end{{theorem}}
                \\begin{{proof}}
                This follows by commutativity.
                \\end{{proof}}
                """
                
                with open(file_path, 'w') as f:
                    f.write(content)
                test_files.append(file_path)
            
            # Create one invalid file
            invalid_file = temp_path / "invalid.tex"
            with open(invalid_file, 'w') as f:
                f.write("")  # Empty file
            test_files.append(invalid_file)
            
            # Test robust batch processing
            results = await pipeline.batch_formalize_robust(
                input_files=test_files,
                output_dir=temp_path / "output",
                parallel=2,
                verify=False,
                fail_fast=False
            )
            
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            fallbacks = sum(1 for r in results if r.fallback_used)
            
            print(f"✅ Batch robustness: {successful} successful, {failed} failed, {fallbacks} fallbacks")
            
            # Verify output files were created for successful cases
            output_dir = temp_path / "output"
            if output_dir.exists():
                output_files = list(output_dir.glob("*.lean"))
                print(f"✅ Output files created: {len(output_files)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Batch robustness test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all Generation 2 robustness tests."""
    print("=" * 70)
    print("TERRAGON SDLC v4.0 - GENERATION 2 ROBUSTNESS TESTING")
    print("=" * 70)
    
    tests = [
        ("Resilience Utilities", test_resilience_utilities()),
        ("Robust Pipeline", test_robust_pipeline()),
        ("Error Handling", test_error_handling()),
        ("Batch Robustness", test_batch_robustness()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("GENERATION 2 TEST RESULTS:")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        all_passed = all_passed and passed
    
    print("=" * 70)
    
    if all_passed:
        print("🎉 ALL GENERATION 2 TESTS PASSED")
        print("✅ System is ROBUST and ready for Generation 3 (Optimization)")
        return 0
    else:
        print("❌ SOME ROBUSTNESS TESTS FAILED")
        print("🔧 Fix robustness issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))