"""Performance benchmarks and load testing."""

import pytest
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock
import statistics

from autoformalize.core.pipeline import FormalizationPipeline
from tests.fixtures import ALL_THEOREMS


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for the formalization system."""
    
    @pytest.fixture
    def performance_pipeline(self, mock_llm_client, mock_proof_assistant):
        """Create a pipeline configured for performance testing."""
        # Mock fast responses
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem perf_test : True := trivial"
        mock_proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        return FormalizationPipeline(
            target_system="lean4",
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant,
            use_cache=False
        )
    
    def test_single_theorem_latency(self, performance_pipeline):
        """Benchmark latency for single theorem processing."""
        theorem = ALL_THEOREMS["quadratic_formula"]["latex"]
        
        # Warm up
        performance_pipeline.formalize(theorem)
        
        # Benchmark
        start_time = time.time()
        result = performance_pipeline.formalize(theorem)
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert result["success"] == True
        assert latency < 1.0  # Should complete in under 1 second
        print(f"Single theorem latency: {latency:.3f}s")
    
    def test_batch_processing_throughput(self, performance_pipeline):
        """Benchmark throughput for batch processing."""
        theorems = [t["latex"] for t in ALL_THEOREMS.values()]
        batch_sizes = [1, 5, 10, 20]
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            batch = theorems[:batch_size]
            
            start_time = time.time()
            results = performance_pipeline.formalize_batch(batch)
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = len(results) / duration
            throughput_results[batch_size] = throughput
            
            assert all(r["success"] for r in results)
            print(f"Batch size {batch_size}: {throughput:.2f} theorems/sec")
        
        # Throughput should generally increase with batch size
        assert throughput_results[10] >= throughput_results[1]
    
    def test_memory_usage_scaling(self, performance_pipeline):
        """Test memory usage scaling with input size."""
        process = psutil.Process(os.getpid())
        
        # Test with different input sizes
        base_theorem = ALL_THEOREMS["binomial_theorem"]["latex"]
        memory_usage = {}
        
        for multiplier in [1, 5, 10, 20]:
            # Create larger input by repeating theorem
            large_input = base_theorem * multiplier
            
            # Measure memory before
            initial_memory = process.memory_info().rss
            
            # Process large input
            result = performance_pipeline.formalize(large_input)
            
            # Measure memory after
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            memory_usage[multiplier] = memory_increase
            
            assert result["success"] == True
            print(f"Input size {multiplier}x: {memory_increase / 1024 / 1024:.1f}MB increase")
        
        # Memory usage should scale reasonably (not exponentially)
        assert memory_usage[20] < memory_usage[1] * 30  # Less than 30x increase
    
    def test_concurrent_processing(self, mock_llm_client, mock_proof_assistant):
        """Test concurrent theorem processing."""
        # Create multiple pipeline instances
        def create_pipeline():
            mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
                "theorem concurrent : True := trivial"
            mock_proof_assistant.verify_proof.return_value = {
                "success": True,
                "errors": [],
                "warnings": []
            }
            return FormalizationPipeline(
                target_system="lean4",
                llm_client=mock_llm_client,
                proof_assistant=mock_proof_assistant
            )
        
        theorems = list(ALL_THEOREMS.values())
        num_workers = 4
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for theorem in theorems:
                pipeline = create_pipeline()
                future = executor.submit(pipeline.formalize, theorem["latex"])
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = len(results) / duration
        
        assert all(r["success"] for r in results)
        assert len(results) == len(theorems)
        print(f"Concurrent processing: {throughput:.2f} theorems/sec with {num_workers} workers")
    
    def test_correction_performance(self, mock_llm_client, mock_proof_assistant):
        """Benchmark self-correction performance."""
        from autoformalize.core.correction import SelfCorrectingPipeline
        
        # Mock correction scenario: fail twice, then succeed
        mock_proof_assistant.verify_proof.side_effect = [
            {"success": False, "errors": ["error 1"], "warnings": []},
            {"success": False, "errors": ["error 2"], "warnings": []}, 
            {"success": True, "errors": [], "warnings": []}
        ]
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem corrected : True := trivial"
        
        pipeline = SelfCorrectingPipeline(
            target_system="lean4",
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant,
            max_correction_rounds=5
        )
        
        theorem = ALL_THEOREMS["euclid_infinitude_primes"]["latex"]
        
        start_time = time.time()
        result = pipeline.formalize_with_feedback(theorem)
        end_time = time.time()
        
        duration = end_time - start_time
        
        assert result["success"] == True
        assert result["correction_rounds"] == 2
        assert duration < 5.0  # Should complete corrections quickly
        print(f"Self-correction with 2 rounds: {duration:.3f}s")
    
    @pytest.mark.slow
    def test_load_testing(self, performance_pipeline):
        """Load testing with sustained high throughput."""
        theorem = ALL_THEOREMS["quadratic_formula"]["latex"]
        num_requests = 100
        
        latencies = []
        
        start_time = time.time()
        
        for i in range(num_requests):
            request_start = time.time()
            result = performance_pipeline.formalize(theorem)
            request_end = time.time()
            
            latencies.append(request_end - request_start)
            assert result["success"] == True
        
        end_time = time.time()
        
        total_duration = end_time - start_time
        throughput = num_requests / total_duration
        
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        print(f"Load test results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Average latency: {avg_latency:.3f}s")
        print(f"  95th percentile latency: {p95_latency:.3f}s")
        
        # Performance assertions
        assert throughput > 10  # At least 10 requests per second
        assert avg_latency < 0.5  # Average latency under 500ms
        assert p95_latency < 1.0  # 95th percentile under 1 second
    
    def test_cache_performance_impact(self, mock_llm_client, mock_proof_assistant, temp_dir):
        """Test performance impact of caching."""
        cache_dir = temp_dir / "perf_cache"
        cache_dir.mkdir()
        
        # Pipeline without cache
        pipeline_no_cache = FormalizationPipeline(
            target_system="lean4",
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant,
            use_cache=False
        )
        
        # Pipeline with cache
        pipeline_with_cache = FormalizationPipeline(
            target_system="lean4", 
            llm_client=mock_llm_client,
            proof_assistant=mock_proof_assistant,
            use_cache=True,
            cache_dir=cache_dir
        )
        
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = \
            "theorem cached : True := trivial"
        mock_proof_assistant.verify_proof.return_value = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        theorem = ALL_THEOREMS["fundamental_theorem_arithmetic"]["latex"]
        
        # Time without cache (first call)
        start_time = time.time()
        result1 = pipeline_no_cache.formalize(theorem)
        no_cache_time = time.time() - start_time
        
        # Time with cache (first call - populates cache)
        start_time = time.time()
        result2 = pipeline_with_cache.formalize(theorem)
        first_cache_time = time.time() - start_time
        
        # Time with cache (second call - uses cache)
        mock_llm_client.reset_mock()
        start_time = time.time()
        result3 = pipeline_with_cache.formalize(theorem)
        cached_time = time.time() - start_time
        
        assert result1["success"] == True
        assert result2["success"] == True
        assert result3["success"] == True
        
        print(f"Performance comparison:")
        print(f"  No cache: {no_cache_time:.3f}s")
        print(f"  First cache call: {first_cache_time:.3f}s")
        print(f"  Cached call: {cached_time:.3f}s")
        
        # Cached call should be significantly faster
        assert cached_time < first_cache_time * 0.5  # At least 50% faster
        
        # LLM should not be called for cached result
        assert mock_llm_client.chat.completions.create.call_count == 0


@pytest.mark.performance
class TestResourceUtilization:
    """Test system resource utilization patterns."""
    
    def test_cpu_utilization(self, performance_pipeline):
        """Monitor CPU utilization during processing."""
        import psutil
        
        theorem = ALL_THEOREMS["binomial_theorem"]["latex"]
        
        # Monitor CPU usage
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        start_time = time.time()
        result = performance_pipeline.formalize(theorem)
        end_time = time.time()
        
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        duration = end_time - start_time
        
        assert result["success"] == True
        print(f"CPU utilization: {cpu_percent_before}% -> {cpu_percent_after}%")
        print(f"Processing time: {duration:.3f}s")
        
        # CPU usage should be reasonable
        assert cpu_percent_after < 90  # Should not max out CPU
    
    def test_memory_leak_detection(self, performance_pipeline):
        """Test for memory leaks over multiple operations."""
        import gc
        
        process = psutil.Process(os.getpid())
        theorem = ALL_THEOREMS["quadratic_formula"]["latex"]
        
        # Record initial memory
        gc.collect()  # Force garbage collection
        initial_memory = process.memory_info().rss
        
        # Process many theorems
        for i in range(50):
            result = performance_pipeline.formalize(theorem)
            assert result["success"] == True
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Record final memory
        gc.collect()
        final_memory = process.memory_info().rss
        
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / 1024 / 1024
        
        print(f"Memory increase after 50 operations: {memory_increase_mb:.1f}MB")
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase_mb < 50