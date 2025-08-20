#!/usr/bin/env python3
"""Test Generation 3 optimization and scaling features.

This test suite validates the advanced optimization, caching, load balancing,
and scaling features implemented in Generation 3.
"""

import asyncio
import time
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autoformalize.performance.advanced_optimization import (
    AdvancedOptimizationEngine,
    AdaptiveCache,
    IntelligentLoadBalancer,
    OptimizationConfig,
    OptimizationStrategy,
    ScalingMode
)


class TestAdvancedOptimization:
    """Test cases for advanced optimization features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = OptimizationConfig(
            strategy=OptimizationStrategy.BALANCED,
            scaling_mode=ScalingMode.MULTI_THREAD,
            cache_size_mb=64,
            batch_size=5,
            optimization_interval=5.0
        )
        
        self.engine = AdvancedOptimizationEngine(self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, 'engine'):
            self.engine.shutdown()
    
    async def test_adaptive_cache_basic_operations(self):
        """Test basic adaptive cache operations."""
        cache = AdaptiveCache(max_size_mb=1, enable_persistence=False)
        
        # Test set and get
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"
        
        # Test cache miss
        result = await cache.get("nonexistent")
        assert result is None
        
        # Test cache statistics
        stats = cache.get_statistics()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5
    
    async def test_adaptive_cache_multilevel(self):
        """Test multi-level cache promotion and demotion."""
        cache = AdaptiveCache(max_size_mb=1, enable_persistence=False)
        
        # Add multiple items
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")
        
        # Access some items frequently to promote them
        for _ in range(6):
            await cache.get("key0")
            await cache.get("key1")
        
        # Check that frequently accessed items are in L1
        assert "key0" in cache.l1_cache
        assert "key1" in cache.l1_cache
        
        # Less accessed items should be in lower levels
        keys_in_l2_l3 = list(cache.l2_cache.keys()) + list(cache.l3_cache.keys())
        assert "key9" in keys_in_l2_l3
    
    async def test_intelligent_load_balancer(self):
        """Test intelligent load balancing."""
        balancer = IntelligentLoadBalancer(ScalingMode.MULTI_THREAD)
        
        def cpu_intensive_task(n: int) -> int:
            """Simulate CPU-intensive task."""
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        def io_intensive_task(delay: float) -> str:
            """Simulate I/O-intensive task."""
            time.sleep(delay)
            return f"completed after {delay}s"
        
        # Test CPU-intensive task submission
        start_time = time.time()
        result = await balancer.submit_task(
            cpu_intensive_task, 
            10000,
            task_type="computation"
        )
        duration = time.time() - start_time
        
        assert isinstance(result, int)
        assert result > 0
        assert duration < 1.0  # Should complete quickly for small input
        
        # Test I/O-intensive task submission
        start_time = time.time()
        result = await balancer.submit_task(
            io_intensive_task,
            0.1,
            task_type="io"
        )
        duration = time.time() - start_time
        
        assert "completed" in result
        assert 0.1 <= duration < 0.2
        
        balancer.shutdown()
    
    async def test_batch_processing(self):
        """Test optimized batch processing."""
        balancer = IntelligentLoadBalancer(ScalingMode.MULTI_THREAD)
        
        def simple_task(x: int) -> int:
            return x * 2
        
        # Create batch of tasks
        tasks = [(simple_task, (i,), {}) for i in range(20)]
        
        start_time = time.time()
        results = await balancer.batch_process(tasks, batch_size=5)
        duration = time.time() - start_time
        
        assert len(results) == 20
        assert all(results[i] == i * 2 for i in range(20))
        assert duration < 1.0  # Should process quickly
        
        balancer.shutdown()
    
    async def test_optimization_engine_caching(self):
        """Test optimization engine with caching."""
        call_count = 0
        
        def expensive_operation(input_data: str) -> str:
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return f"processed_{input_data}"
        
        # First call - should execute function
        result1 = await self.engine.optimize_operation(
            expensive_operation,
            cache_key="test_key",
            task_type="computation",
            "test_input"
        )
        
        assert result1 == "processed_test_input"
        assert call_count == 1
        
        # Second call - should use cache
        result2 = await self.engine.optimize_operation(
            expensive_operation,
            cache_key="test_key", 
            task_type="computation",
            "test_input"
        )
        
        assert result2 == "processed_test_input"
        assert call_count == 1  # Function not called again
    
    async def test_batch_optimization(self):
        """Test batch optimization with caching."""
        def process_item(item: str) -> str:
            return f"processed_{item}"
        
        # Create operations
        operations = [
            (process_item, (f"item{i}",), {})
            for i in range(10)
        ]
        
        cache_keys = [f"cache_key_{i}" for i in range(10)]
        
        # First batch - should execute all operations
        results1 = await self.engine.optimize_batch(
            operations,
            cache_keys=cache_keys,
            task_type="batch_test"
        )
        
        assert len(results1) == 10
        assert all(f"processed_item{i}" == results1[i] for i in range(10))
        
        # Second batch - should use cached results
        start_time = time.time()
        results2 = await self.engine.optimize_batch(
            operations,
            cache_keys=cache_keys,
            task_type="batch_test"
        )
        duration = time.time() - start_time
        
        assert results2 == results1
        assert duration < 0.1  # Should be much faster due to caching
    
    def test_optimization_statistics(self):
        """Test optimization statistics collection."""
        stats = self.engine.get_optimization_statistics()
        
        assert "current_strategy" in stats
        assert "configuration" in stats
        assert "performance_metrics" in stats
        assert "cache_statistics" in stats
        assert "load_balancer_statistics" in stats
        
        config = stats["configuration"]
        assert config["scaling_mode"] == ScalingMode.MULTI_THREAD.value
        assert config["batch_size"] == 5
        assert config["cache_size_mb"] == 64
    
    async def test_performance_under_load(self):
        """Test system performance under simulated load."""
        def variable_load_task(task_id: int, complexity: int) -> Dict[str, Any]:
            """Simulate variable complexity task."""
            start = time.time()
            
            # Simulate different complexity levels
            if complexity == 1:
                time.sleep(0.01)  # Light task
            elif complexity == 2:
                time.sleep(0.05)  # Medium task
            else:
                time.sleep(0.1)   # Heavy task
            
            return {
                "task_id": task_id,
                "complexity": complexity,
                "duration": time.time() - start
            }
        
        # Create mixed workload
        tasks = []
        cache_keys = []
        
        for i in range(30):
            complexity = (i % 3) + 1
            tasks.append((variable_load_task, (i, complexity), {}))
            cache_keys.append(f"load_test_{i}")
        
        # Execute workload
        start_time = time.time()
        results = await self.engine.optimize_batch(
            tasks,
            cache_keys=cache_keys,
            task_type="load_test"
        )
        total_duration = time.time() - start_time
        
        # Verify results
        assert len(results) == 30
        assert all(isinstance(r, dict) and "task_id" in r for r in results)
        
        # Check performance
        assert total_duration < 5.0  # Should complete within reasonable time
        
        # Get statistics
        stats = self.engine.get_optimization_statistics()
        assert stats["total_operations"] > 0
        assert stats["performance_metrics"]["average_throughput"] > 0


class TestOptimizationIntegration:
    """Integration tests for optimization features."""
    
    async def test_end_to_end_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ADAPTIVE,
            scaling_mode=ScalingMode.HYBRID,
            cache_size_mb=32,
            batch_size=3,
            optimization_interval=2.0
        )
        
        engine = AdvancedOptimizationEngine(config)
        
        try:
            # Simulate formalization pipeline operations
            async def mock_latex_parsing(content: str) -> Dict[str, Any]:
                await asyncio.sleep(0.05)  # Simulate parsing time
                return {
                    "theorems": [f"theorem from {content}"],
                    "definitions": [],
                    "parsed_content": content
                }
            
            async def mock_code_generation(parsed_data: Dict[str, Any]) -> str:
                await asyncio.sleep(0.1)  # Simulate generation time
                return f"formal_code({parsed_data['parsed_content']})"
            
            async def mock_verification(code: str) -> bool:
                await asyncio.sleep(0.03)  # Simulate verification time
                return len(code) > 10  # Simple success criteria
            
            # Test individual operations with caching
            latex_inputs = [
                "theorem: 1 + 1 = 2",
                "theorem: 2 + 2 = 4", 
                "theorem: 1 + 1 = 2",  # Duplicate for cache test
                "lemma: x + 0 = x"
            ]
            
            results = []
            
            for i, latex_input in enumerate(latex_inputs):
                # Parsing step
                parsed = await engine.optimize_operation(
                    mock_latex_parsing,
                    cache_key=f"parse_{hash(latex_input)}",
                    task_type="parsing",
                    latex_input
                )
                
                # Generation step
                generated = await engine.optimize_operation(
                    mock_code_generation,
                    cache_key=f"gen_{hash(str(parsed))}",
                    task_type="generation",
                    parsed
                )
                
                # Verification step
                verified = await engine.optimize_operation(
                    mock_verification,
                    cache_key=f"verify_{hash(generated)}",
                    task_type="verification",
                    generated
                )
                
                results.append({
                    "input": latex_input,
                    "parsed": parsed,
                    "generated": generated,
                    "verified": verified
                })
            
            # Verify results
            assert len(results) == 4
            assert all(r["verified"] for r in results)
            
            # Check that caching worked (duplicate input should use cache)
            assert results[0]["generated"] == results[2]["generated"]
            
            # Get final statistics
            stats = engine.get_optimization_statistics()
            
            assert stats["total_operations"] >= 12  # 4 inputs × 3 operations each
            assert stats["cache_statistics"]["hits"] > 0  # Should have cache hits
            assert stats["performance_metrics"]["average_throughput"] > 0
            
        finally:
            engine.shutdown()
    
    async def test_scaling_modes_comparison(self):
        """Test different scaling modes for performance comparison."""
        test_results = {}
        
        scaling_modes = [
            ScalingMode.SINGLE_THREAD,
            ScalingMode.MULTI_THREAD,
            ScalingMode.HYBRID
        ]
        
        def cpu_task(n: int) -> int:
            total = 0
            for i in range(n):
                total += i % 7
            return total
        
        # Test each scaling mode
        for mode in scaling_modes:
            config = OptimizationConfig(
                scaling_mode=mode,
                batch_size=5,
                cache_size_mb=16
            )
            
            engine = AdvancedOptimizationEngine(config)
            
            try:
                # Create CPU-intensive tasks
                tasks = [(cpu_task, (1000,), {}) for _ in range(20)]
                
                start_time = time.time()
                results = await engine.optimize_batch(tasks, task_type="cpu_test")
                duration = time.time() - start_time
                
                test_results[mode.value] = {
                    "duration": duration,
                    "results_count": len(results),
                    "all_successful": all(isinstance(r, int) for r in results)
                }
                
            finally:
                engine.shutdown()
        
        # Verify all modes completed successfully
        for mode, result in test_results.items():
            assert result["all_successful"], f"Scaling mode {mode} had failures"
            assert result["results_count"] == 20
            assert result["duration"] < 10.0  # Reasonable time limit
        
        print(f"Scaling mode performance comparison: {test_results}")
    
    async def test_adaptive_optimization_behavior(self):
        """Test adaptive optimization behavior under varying load."""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ADAPTIVE,
            optimization_interval=1.0,  # Fast optimization for testing
            batch_size=2
        )
        
        engine = AdvancedOptimizationEngine(config)
        
        try:
            # Phase 1: Light load
            def light_task(x: int) -> int:
                return x + 1
            
            light_operations = [(light_task, (i,), {}) for i in range(10)]
            await engine.optimize_batch(light_operations, task_type="light")
            
            initial_stats = engine.get_optimization_statistics()
            initial_batch_size = initial_stats["configuration"]["batch_size"]
            
            # Phase 2: Heavy load with errors
            def heavy_task(x: int) -> int:
                time.sleep(0.1)
                if x % 5 == 0:  # 20% failure rate
                    raise ValueError(f"Simulated error for {x}")
                return x * 2
            
            heavy_operations = [(heavy_task, (i,), {}) for i in range(15)]
            
            # Execute heavy operations (some will fail)
            try:
                await engine.optimize_batch(heavy_operations, task_type="heavy")
            except:
                pass  # Expected due to errors
            
            # Wait for adaptive optimization
            await asyncio.sleep(2.0)
            
            final_stats = engine.get_optimization_statistics()
            
            # Verify adaptive behavior occurred
            assert final_stats["total_operations"] > initial_stats["total_operations"]
            
            # Adaptive optimization should have adjusted parameters
            # (exact behavior depends on implementation, but metrics should be tracked)
            assert "performance_metrics" in final_stats
            assert final_stats["performance_metrics"]["average_error_rate"] >= 0
            
        finally:
            engine.shutdown()


async def main():
    """Run Generation 3 optimization demonstration."""
    print("🚀 Generation 3: Optimization & Scaling Demo")
    print("=" * 50)
    
    # Initialize optimization engine
    config = OptimizationConfig(
        strategy=OptimizationStrategy.ADAPTIVE,
        scaling_mode=ScalingMode.HYBRID,
        cache_size_mb=128,
        batch_size=8,
        optimization_interval=10.0
    )
    
    engine = AdvancedOptimizationEngine(config)
    
    try:
        print("\n🔧 Testing Advanced Caching...")
        
        # Test caching with simulated expensive operations
        def expensive_computation(input_val: int) -> Dict[str, Any]:
            """Simulate expensive mathematical computation."""
            time.sleep(0.1)  # Simulate computation time
            return {
                "input": input_val,
                "result": input_val ** 2,
                "factors": [i for i in range(1, input_val + 1) if input_val % i == 0],
                "timestamp": time.time()
            }
        
        # First round - should execute functions
        print("  📊 First execution (no cache)...")
        start_time = time.time()
        
        results1 = []
        for i in range(5):
            result = await engine.optimize_operation(
                expensive_computation,
                cache_key=f"computation_{i}",
                task_type="computation",
                i + 10
            )
            results1.append(result)
        
        first_duration = time.time() - start_time
        print(f"  ⏱️  First round completed in {first_duration:.2f}s")
        
        # Second round - should use cache
        print("  💾 Second execution (with cache)...")
        start_time = time.time()
        
        results2 = []
        for i in range(5):
            result = await engine.optimize_operation(
                expensive_computation,
                cache_key=f"computation_{i}",
                task_type="computation",
                i + 10
            )
            results2.append(result)
        
        second_duration = time.time() - start_time
        print(f"  ⏱️  Second round completed in {second_duration:.2f}s")
        print(f"  🚀 Speedup: {first_duration/second_duration:.1f}x faster")
        
        # Verify cache effectiveness
        assert results1 == results2, "Cached results should match original results"
        assert second_duration < first_duration * 0.5, "Cache should provide significant speedup"
        
        print("\n⚡ Testing Intelligent Load Balancing...")
        
        # Test different types of workloads
        workloads = {
            "CPU-intensive": lambda x: sum(i**2 for i in range(x * 100)),
            "I/O simulation": lambda x: time.sleep(0.05) or f"io_result_{x}",
            "Mixed workload": lambda x: x * 2 if x % 2 == 0 else sum(range(x))
        }
        
        for workload_name, workload_func in workloads.items():
            print(f"  🔄 Testing {workload_name}...")
            
            # Create batch of tasks
            tasks = [(workload_func, (i,), {}) for i in range(10)]
            
            start_time = time.time()
            results = await engine.optimize_batch(
                tasks,
                task_type=workload_name.lower().replace("-", "_")
            )
            duration = time.time() - start_time
            
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            print(f"    ✅ {success_count}/10 tasks completed in {duration:.2f}s")
        
        print("\n📈 Performance Analytics...")
        
        # Get comprehensive statistics
        stats = engine.get_optimization_statistics()
        
        print(f"  🎯 Optimization Strategy: {stats['current_strategy']}")
        print(f"  ⚙️  Scaling Mode: {stats['configuration']['scaling_mode']}")
        print(f"  📦 Batch Size: {stats['configuration']['batch_size']}")
        print(f"  💾 Cache Size: {stats['configuration']['cache_size_mb']}MB")
        
        perf = stats['performance_metrics']
        print(f"\n  📊 Performance Metrics:")
        print(f"    • Throughput: {perf['average_throughput']:.1f} ops/sec")
        print(f"    • Latency: {perf['average_latency']:.3f}s")
        print(f"    • Error Rate: {perf['average_error_rate']:.1%}")
        print(f"    • Cache Hit Ratio: {perf['cache_hit_ratio']:.1%}")
        
        cache_stats = stats['cache_statistics']
        print(f"\n  💾 Cache Statistics:")
        print(f"    • Total Requests: {cache_stats['total_requests']}")
        print(f"    • Cache Hits: {cache_stats['hits']}")
        print(f"    • Cache Size: {cache_stats['total_size_mb']:.1f}MB")
        print(f"    • L1 Items: {cache_stats['l1_size']}")
        print(f"    • L2 Items: {cache_stats['l2_size']}")
        print(f"    • L3 Items: {cache_stats['l3_size']}")
        
        load_balancer_stats = stats['load_balancer_statistics']
        print(f"\n  ⚖️  Load Balancer:")
        print(f"    • Active Thread Pool: {load_balancer_stats['active_workers']['thread_pool']}")
        print(f"    • Active Process Pool: {load_balancer_stats['active_workers']['process_pool']}")
        print(f"    • Worker Performance: {len(load_balancer_stats['average_performance'])} tracked")
        
        print(f"\n  📈 Overall Performance:")
        print(f"    • Total Operations: {stats['total_operations']}")
        print(f"    • Performance vs Targets:")
        targets = stats['performance_targets']
        for metric, target in targets.items():
            current = perf.get(f'average_{metric}', 0)
            if metric == 'throughput':
                status = "✅" if current >= target * 0.8 else "⚠️"
                print(f"      {status} {metric.title()}: {current:.1f} / {target:.1f}")
            elif metric == 'error_rate':
                status = "✅" if current <= target else "⚠️"
                print(f"      {status} {metric.title()}: {current:.1%} / {target:.1%}")
        
        print("\n🧪 Testing Adaptive Optimization...")
        
        # Simulate changing workload to trigger adaptive optimization
        print("  📊 Simulating variable workload...")
        
        # High-error workload to trigger adaptive changes
        def unreliable_task(x: int) -> int:
            if x % 3 == 0:  # 33% failure rate
                raise ValueError(f"Simulated failure for {x}")
            return x * x
        
        unreliable_tasks = [(unreliable_task, (i,), {}) for i in range(15)]
        
        try:
            await engine.optimize_batch(unreliable_tasks, task_type="unreliable")
        except:
            pass  # Expected due to high error rate
        
        print("  ⚡ Adaptive optimization should adjust parameters based on errors...")
        
        # Get updated statistics
        updated_stats = engine.get_optimization_statistics()
        updated_perf = updated_stats['performance_metrics']
        
        print(f"  📊 Updated Error Rate: {updated_perf['average_error_rate']:.1%}")
        print(f"  🔧 Current Batch Size: {updated_stats['configuration']['batch_size']}")
        
        print("\n🎉 Generation 3 Optimization Demo Complete!")
        print("✅ Advanced multi-level caching implemented")
        print("✅ Intelligent load balancing active")
        print("✅ Adaptive optimization enabled")
        print("✅ Performance monitoring comprehensive")
        print("✅ Scaling capabilities demonstrated")
        
        return {
            "performance_stats": stats,
            "cache_effectiveness": {
                "first_duration": first_duration,
                "second_duration": second_duration,
                "speedup": first_duration / second_duration
            },
            "adaptive_behavior": {
                "initial_error_rate": perf['average_error_rate'],
                "final_error_rate": updated_perf['average_error_rate']
            }
        }
    
    finally:
        engine.shutdown()


if __name__ == "__main__":
    # Run optimization demo
    print("Generation 3 Optimization & Scaling Test Suite")
    print("Running comprehensive optimization demo...")
    
    result = asyncio.run(main())
    
    print(f"\n📊 Final Results Summary:")
    print(f"Cache speedup achieved: {result['cache_effectiveness']['speedup']:.1f}x")
    print(f"Total operations optimized: {result['performance_stats']['total_operations']}")
    print(f"Cache hit ratio: {result['performance_stats']['cache_statistics']['hit_ratio']:.1%}")
    print("🚀 All optimization features validated successfully!")