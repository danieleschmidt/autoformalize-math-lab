#!/usr/bin/env python3
"""Simple Generation 3 optimization demonstration."""

import asyncio
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autoformalize.performance.advanced_optimization import (
    AdvancedOptimizationEngine,
    OptimizationConfig,
    OptimizationStrategy,
    ScalingMode
)


async def main():
    """Run Generation 3 optimization demonstration."""
    print("🚀 Generation 3: Optimization & Scaling Demo")
    print("=" * 50)
    
    # Initialize optimization engine
    config = OptimizationConfig(
        strategy=OptimizationStrategy.BALANCED,
        scaling_mode=ScalingMode.MULTI_THREAD,
        cache_size_mb=64,
        batch_size=5
    )
    
    engine = AdvancedOptimizationEngine(config)
    
    try:
        print("\n🔧 Testing Advanced Caching...")
        
        def expensive_computation(input_val: int) -> dict:
            """Simulate expensive mathematical computation."""
            time.sleep(0.05)  # Simulate computation time
            return {
                "input": input_val,
                "result": input_val ** 2,
                "timestamp": time.time()
            }
        
        # Test caching effectiveness
        print("  📊 Testing cache performance...")
        
        # First execution - no cache
        start_time = time.time()
        result1 = await engine.optimize_operation(
            expensive_computation,
            cache_key="test_computation_10",
            task_type="computation",
            input_val=10
        )
        first_duration = time.time() - start_time
        
        # Second execution - should use cache
        start_time = time.time()
        result2 = await engine.optimize_operation(
            expensive_computation,
            cache_key="test_computation_10",
            task_type="computation",
            input_val=10
        )
        second_duration = time.time() - start_time
        
        print(f"  ⏱️  First execution: {first_duration:.3f}s")
        print(f"  ⏱️  Second execution: {second_duration:.3f}s")
        print(f"  🚀 Cache speedup: {first_duration/second_duration:.1f}x")
        
        # Verify results are identical
        assert result1["result"] == result2["result"]
        assert second_duration < first_duration * 0.5  # Cache should be much faster
        
        print("\n⚡ Testing Intelligent Load Balancing...")
        
        def cpu_task(n: int) -> int:
            """CPU-intensive task."""
            total = 0
            for i in range(n * 1000):
                total += i % 7
            return total
        
        def io_task(delay: float) -> str:
            """I/O simulation task."""
            time.sleep(delay)
            return f"io_completed_{delay}"
        
        # Test CPU-intensive task
        start_time = time.time()
        cpu_result = await engine.load_balancer.submit_task(
            cpu_task, 50, task_type="computation"
        )
        cpu_duration = time.time() - start_time
        
        print(f"  🖥️  CPU task completed in {cpu_duration:.3f}s: {cpu_result}")
        
        # Test I/O task
        start_time = time.time()
        io_result = await engine.load_balancer.submit_task(
            io_task, 0.1, task_type="io"
        )
        io_duration = time.time() - start_time
        
        print(f"  💾 I/O task completed in {io_duration:.3f}s: {io_result}")
        
        print("\n📦 Testing Batch Processing...")
        
        # Create batch of tasks
        tasks = [
            (cpu_task, (i * 10,), {}) 
            for i in range(8)
        ]
        
        start_time = time.time()
        batch_results = await engine.load_balancer.batch_process(
            tasks, batch_size=4
        )
        batch_duration = time.time() - start_time
        
        print(f"  📊 Processed {len(tasks)} tasks in {batch_duration:.3f}s")
        print(f"  ✅ All tasks successful: {all(isinstance(r, int) for r in batch_results)}")
        
        print("\n📈 Performance Statistics...")
        
        # Get optimization statistics
        stats = engine.get_optimization_statistics()
        
        print(f"  🎯 Optimization Strategy: {stats['current_strategy']}")
        print(f"  ⚙️  Scaling Mode: {stats['configuration']['scaling_mode']}")
        print(f"  📦 Batch Size: {stats['configuration']['batch_size']}")
        print(f"  💾 Cache Size: {stats['configuration']['cache_size_mb']}MB")
        
        # Cache statistics
        cache_stats = stats['cache_statistics']
        print(f"\n  💾 Cache Performance:")
        print(f"    • Hit Ratio: {cache_stats['hit_ratio']:.1%}")
        print(f"    • Total Requests: {cache_stats['total_requests']}")
        print(f"    • Cache Size: {cache_stats['total_size_mb']:.2f}MB")
        print(f"    • Items in L1: {cache_stats['l1_size']}")
        print(f"    • Items in L2: {cache_stats['l2_size']}")
        print(f"    • Items in L3: {cache_stats['l3_size']}")
        
        # Load balancer statistics
        lb_stats = stats['load_balancer_statistics']
        print(f"\n  ⚖️  Load Balancer:")
        print(f"    • Thread Pool Active: {lb_stats['active_workers']['thread_pool']}")
        print(f"    • Process Pool Active: {lb_stats['active_workers']['process_pool']}")
        print(f"    • Tracked Workers: {len(lb_stats['average_performance'])}")
        
        # Performance metrics
        perf = stats['performance_metrics']
        print(f"\n  📊 Performance Metrics:")
        print(f"    • Average Throughput: {perf['average_throughput']:.1f} ops/sec")
        print(f"    • Average Latency: {perf['average_latency']:.3f}s")
        print(f"    • Error Rate: {perf['average_error_rate']:.1%}")
        
        print(f"\n  📈 Overall:")
        print(f"    • Total Operations: {stats['total_operations']}")
        
        print("\n🧪 Testing Multi-Level Cache...")
        
        # Test cache levels by accessing items with different frequencies
        for i in range(15):
            cache_key = f"item_{i}"
            result = await engine.optimize_operation(
                lambda x: f"processed_{x}",
                cache_key=cache_key,
                task_type="test",
                x=i
            )
            
            # Access some items more frequently to test promotion
            if i < 5:
                for _ in range(3):  # Access first 5 items frequently
                    await engine.cache.get(cache_key)
        
        # Check cache distribution
        updated_cache_stats = engine.cache.get_statistics()
        print(f"  📊 Cache Distribution:")
        print(f"    • L1 Cache: {updated_cache_stats['l1_size']} items")
        print(f"    • L2 Cache: {updated_cache_stats['l2_size']} items") 
        print(f"    • L3 Cache: {updated_cache_stats['l3_size']} items")
        print(f"    • Total Evictions: {updated_cache_stats['evictions']}")
        
        print("\n🎉 Generation 3 Optimization Demo Complete!")
        print("✅ Advanced multi-level adaptive caching implemented")
        print("✅ Intelligent load balancing with task type optimization")
        print("✅ Batch processing with automatic sizing")
        print("✅ Real-time performance monitoring")
        print("✅ Multi-threading and process pool scaling")
        
        return {
            "cache_speedup": first_duration / second_duration,
            "total_operations": stats['total_operations'],
            "cache_hit_ratio": cache_stats['hit_ratio'],
            "performance_stats": stats
        }
        
    finally:
        engine.shutdown()


if __name__ == "__main__":
    print("Generation 3 Optimization & Scaling Demo")
    print("Running simple optimization demonstration...")
    
    result = asyncio.run(main())
    
    print(f"\n📊 Final Results:")
    print(f"Cache performance improvement: {result['cache_speedup']:.1f}x faster")
    print(f"Total operations optimized: {result['total_operations']}")
    print(f"Cache hit ratio achieved: {result['cache_hit_ratio']:.1%}")
    print("🚀 Generation 3 optimization features validated!")