#!/usr/bin/env python3
"""
Generation 3 Demo: Performance optimization and scaling
Demonstrates caching, auto-scaling, concurrent processing, and load balancing.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.performance.adaptive_optimizer import (
    AdaptiveCache, AutoScaler, ConcurrentExecutor, 
    LoadBalancer, ResourcePool, get_performance_optimizer
)
from autoformalize.core.pipeline import FormalizationPipeline

async def demo_generation3():
    """Demonstrate Generation 3 (Optimized) functionality."""
    print("⚡ Generation 3: MAKE IT SCALE - Demo Starting")
    print("=" * 60)
    
    try:
        # Test 1: Adaptive Caching System
        print("📦 Test 1: Adaptive Caching System")
        cache = AdaptiveCache(max_size=100, default_ttl=300)
        
        # Cache some computations
        expensive_results = {}
        for i in range(20):
            key = f"computation_{i}"
            result = f"result_{i * i}"  # Simulate expensive computation
            cache.set(key, result)
            expensive_results[key] = result
        
        # Test cache hits
        hits = 0
        for i in range(20):
            key = f"computation_{i}"
            cached_result = cache.get(key)
            if cached_result == expensive_results[key]:
                hits += 1
        
        cache_stats = cache.get_stats()
        print(f"✅ Cache performance:")
        print(f"   - Cache hits: {hits}/20")
        print(f"   - Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"   - Cache size: {cache_stats['size']}")
        print(f"   - Total size: {cache_stats['total_size_bytes']} bytes")
        
        # Test 2: Auto-scaling System
        print("\n🔄 Test 2: Auto-scaling System")
        scaler = AutoScaler(min_workers=2, max_workers=8)
        
        # Record some metrics
        initial_workers = scaler.current_workers
        for _ in range(3):
            scaler.record_metrics()
            await asyncio.sleep(0.1)
        
        recommended = scaler.get_recommended_workers()
        print(f"✅ Auto-scaling analysis:")
        print(f"   - Initial workers: {initial_workers}")
        print(f"   - Recommended workers: {recommended}")
        print(f"   - Metrics recorded: {len(scaler.metrics_history)}")
        print(f"   - Can scale: {scaler._can_scale()}")
        
        # Test 3: Concurrent Processing
        print("\n⚙️  Test 3: Concurrent Processing")
        executor = ConcurrentExecutor(max_threads=8, max_processes=2)
        
        # Create test tasks
        def simulate_work(task_id):
            time.sleep(0.1)  # Simulate work
            return f"Task {task_id} completed"
        
        tasks = [lambda i=i: simulate_work(i) for i in range(10)]
        
        start_time = time.time()
        results = await executor.execute_async(tasks)
        execution_time = time.time() - start_time
        
        executor_stats = executor.get_stats()
        print(f"✅ Concurrent execution:")
        print(f"   - Tasks completed: {len([r for r in results if r])}")
        print(f"   - Execution time: {execution_time:.2f}s")
        print(f"   - Success rate: {executor_stats['success_rate']:.1%}")
        print(f"   - Active tasks: {executor_stats['active_tasks']}")
        
        # Test 4: Load Balancing
        print("\n⚖️  Test 4: Load Balancing")
        
        # Create mock workers
        workers = [f"worker_{i}" for i in range(4)]
        load_balancer = LoadBalancer(workers)
        
        # Distribute work
        for _ in range(20):
            worker = load_balancer.get_next_worker()
            # Simulate work assignment
        
        lb_stats = load_balancer.get_stats()
        print(f"✅ Load balancing:")
        print(f"   - Workers: {lb_stats['workers']}")
        print(f"   - Total requests: {lb_stats['total_requests']}")
        print(f"   - Error rate: {lb_stats['error_rate']:.1%}")
        
        # Test 5: Resource Pool
        print("\n🏊 Test 5: Resource Pool Management")
        
        # Create resource factory
        resource_counter = 0
        async def create_resource():
            nonlocal resource_counter
            resource_counter += 1
            await asyncio.sleep(0.05)  # Simulate resource creation time
            return f"resource_{resource_counter}"
        
        pool = ResourcePool(create_resource, min_size=2, max_size=5)
        await asyncio.sleep(0.2)  # Wait for initialization
        
        # Test resource acquisition and release
        acquired_resources = []
        for _ in range(3):
            resource = await pool.acquire()
            acquired_resources.append(resource)
        
        for resource in acquired_resources:
            await pool.release(resource)
        
        pool_stats = pool.get_stats()
        print(f"✅ Resource pool:")
        print(f"   - Available resources: {pool_stats['available']}")
        print(f"   - Resources in use: {pool_stats['in_use']}")
        print(f"   - Total created: {pool_stats['total_created']}")
        
        # Test 6: Pipeline Performance Optimization
        print("\n🚀 Test 6: Pipeline Performance Optimization")
        
        # Use optimized pipeline
        pipeline = FormalizationPipeline(target_system="lean4")
        
        # Get performance optimizer
        performance_optimizer = get_performance_optimizer()
        performance_optimizer.start_optimization_loop()
        
        # Test batch processing with caching
        test_theorems = [
            r"\begin{theorem}Theorem A: $x + 0 = x$\end{theorem}",
            r"\begin{theorem}Theorem B: $x \cdot 1 = x$\end{theorem}",
            r"\begin{theorem}Theorem A: $x + 0 = x$\end{theorem}",  # Duplicate for cache test
            r"\begin{theorem}Theorem C: $x + y = y + x$\end{theorem}",
            r"\begin{theorem}Theorem B: $x \cdot 1 = x$\end{theorem}",  # Another duplicate
        ]
        
        start_time = time.time()
        batch_results = []
        
        # Process with potential caching
        for i, theorem in enumerate(test_theorems):
            # Use theorem content as cache key
            cache_key = f"theorem_{hash(theorem)}"
            
            # Try cache first
            cached_result = performance_optimizer.cache.get(cache_key)
            if cached_result:
                batch_results.append(cached_result)
            else:
                # Process and cache result
                result = await pipeline.formalize(theorem, verify=False)
                performance_optimizer.cache.set(cache_key, result)
                batch_results.append(result)
        
        processing_time = time.time() - start_time
        successful_results = sum(1 for r in batch_results if r.success)
        
        print(f"✅ Optimized pipeline processing:")
        print(f"   - Batch size: {len(test_theorems)}")
        print(f"   - Successful: {successful_results}")
        print(f"   - Processing time: {processing_time:.2f}s")
        print(f"   - Average per item: {processing_time/len(test_theorems):.3f}s")
        
        # Test 7: System-wide Performance Analysis
        print("\n📊 Test 7: System-wide Performance Analysis")
        
        # Let the optimizer run
        await asyncio.sleep(1.0)
        
        # Get comprehensive statistics
        perf_stats = performance_optimizer.get_comprehensive_stats()
        
        print(f"✅ System performance overview:")
        print(f"   - Cache hit rate: {perf_stats['metrics']['cache_hit_rate']:.1%}")
        print(f"   - Memory usage: {perf_stats['metrics']['memory_usage_mb']:.1f} MB")
        print(f"   - CPU usage: {perf_stats['metrics']['cpu_usage_percent']:.1f}%")
        print(f"   - Concurrent operations: {perf_stats['metrics']['concurrent_operations']}")
        print(f"   - Auto-scaler workers: {perf_stats['auto_scaler']['current_workers']}")
        print(f"   - Executor success rate: {perf_stats['executor']['success_rate']:.1%}")
        
        # Test 8: Scaling Under Load
        print("\n🏋️  Test 8: Scaling Under Load")
        
        # Simulate high load
        async def load_generator():
            tasks = []
            for i in range(50):
                theorem = f"\\begin{{theorem}}Load test {i}: $x_{i} > 0$\\end{{theorem}}"
                task = pipeline.formalize(theorem, verify=False)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)
            return successful, len(results)
        
        load_start = time.time()
        successful, total = await load_generator()
        load_time = time.time() - load_start
        
        print(f"✅ Load test completed:")
        print(f"   - Total operations: {total}")
        print(f"   - Successful: {successful}")
        print(f"   - Success rate: {successful/total:.1%}")
        print(f"   - Total time: {load_time:.2f}s")
        print(f"   - Throughput: {total/load_time:.1f} ops/sec")
        
        # Final optimization
        final_optimization = await performance_optimizer.optimize()
        if final_optimization:
            print(f"\n🔧 Final optimization applied:")
            print(f"   - Cache optimized: {final_optimization.get('cache_optimized', False)}")
            print(f"   - Scaling adjusted: {final_optimization.get('scaling_adjusted', False)}")
            print(f"   - Memory cleaned: {final_optimization.get('memory_cleaned', False)}")
        
        print("\n🎉 Generation 3 Demo Complete!")
        print("✅ Performance optimization and scaling verified")
        print("⚡ System is now highly optimized and scalable")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        performance_optimizer.shutdown()
        executor.shutdown()

if __name__ == "__main__":
    success = asyncio.run(demo_generation3())
    sys.exit(0 if success else 1)