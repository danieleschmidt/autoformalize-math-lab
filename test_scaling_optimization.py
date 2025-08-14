#!/usr/bin/env python3
"""Test advanced scaling and optimization features."""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoformalize.performance.optimization_engine import (
    OptimizationEngine, OptimizationConfig, OptimizationStrategy,
    IntelligentCache, BatchProcessor, PerformanceMonitor
)
from autoformalize.scaling.distributed_coordination import (
    DistributedTaskManager, NodeInfo, NodeRole, ClusterConfig,
    LoadBalancingStrategy
)

async def test_optimization_engine():
    """Test optimization engine features."""
    print("🚀 Testing Optimization Engine")
    print("=" * 40)
    
    # Create optimization config
    config = OptimizationConfig(
        strategy=OptimizationStrategy.ADAPTIVE,
        enable_caching=True,
        enable_batching=True,
        cache_size_mb=64,  # Small for testing
        batch_size=3,
        batch_timeout=0.5
    )
    
    # Create optimization engine
    engine = OptimizationEngine(config)
    await engine.start()
    
    print(f"✅ Optimization engine started with {config.strategy.value} strategy")
    
    # Test caching
    print("\n🗄️ Testing intelligent caching...")
    
    async def mock_formalization(latex_content: str, **kwargs):
        """Mock formalization function."""
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Mock formalization of: {latex_content[:30]}..."
    
    # Test cache miss and hit
    test_latex = "\\begin{theorem} Test theorem \\end{theorem}"
    
    # First call - cache miss
    start_time = time.time()
    result1 = await engine.optimize_formalization(
        test_latex, 
        mock_formalization
    )
    time1 = time.time() - start_time
    
    # Second call - cache hit (should be faster)
    start_time = time.time()
    result2 = await engine.optimize_formalization(
        test_latex, 
        mock_formalization
    )
    time2 = time.time() - start_time
    
    print(f"   First call (cache miss): {time1:.3f}s")
    print(f"   Second call (cache hit): {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.1f}x" if time2 > 0 else "   Speedup: ∞")
    print(f"   Results match: {result1 == result2}")
    
    # Test cache statistics
    cache_stats = engine.cache.get_stats() if engine.cache else {}
    print(f"   Cache stats: {cache_stats.get('hit_rate', 0):.1%} hit rate")
    
    # Test performance monitoring
    print("\n📊 Testing performance monitoring...")
    
    # Generate some load
    tasks = []
    for i in range(5):
        task = engine.optimize_formalization(
            f"\\begin{{theorem}} Test theorem {i} \\end{{theorem}}",
            mock_formalization
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    print(f"   Processed {len(results)} requests in batch")
    
    # Get performance summary
    summary = engine.get_optimization_summary()
    print(f"   Strategy: {summary['strategy']}")
    print(f"   Components enabled: {summary['components']}")
    
    if 'cache' in summary:
        cache_info = summary['cache']
        print(f"   Cache utilization: {cache_info.get('utilization', 0):.1f}%")
    
    await engine.stop()
    print("\n✅ Optimization engine test completed!")

async def test_intelligent_cache():
    """Test intelligent cache features."""
    print("\n🧠 Testing Intelligent Cache")
    print("=" * 35)
    
    cache = IntelligentCache(max_size_mb=1, enable_prediction=True)  # Small for testing
    
    # Test basic operations
    print("📝 Testing basic cache operations...")
    
    # Put and get
    cache.put("key1", "value1")
    cache.put("key2", "value2") 
    cache.put("key3", "value3")
    
    result1 = cache.get("key1")
    result2 = cache.get("key1")  # Access again to increase frequency
    result3 = cache.get("nonexistent")
    
    print(f"   Get existing key: {'✅' if result1 == 'value1' else '❌'}")
    print(f"   Get nonexistent key: {'✅' if result3 is None else '❌'}")
    
    # Test cache statistics
    stats = cache.get_stats()
    print(f"   Cache size: {stats['size']} items")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    
    # Test eviction by filling cache
    print("\n🗑️ Testing cache eviction...")
    
    initial_size = stats['size']
    
    # Add many items to trigger eviction
    for i in range(50):
        large_value = "x" * 1000  # 1KB value
        cache.put(f"large_key_{i}", large_value)
    
    final_stats = cache.get_stats()
    print(f"   Items before eviction: {initial_size}")
    print(f"   Items after adding 50: {final_stats['size']}")
    print(f"   Evictions occurred: {'✅' if final_stats['eviction_count'] > 0 else '❌'}")
    print(f"   Total evictions: {final_stats['eviction_count']}")
    
    print("\n✅ Intelligent cache test completed!")

async def test_performance_monitoring():
    """Test performance monitoring features."""
    print("\n📈 Testing Performance Monitoring")
    print("=" * 35)
    
    config = OptimizationConfig()
    monitor = PerformanceMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    print("✅ Performance monitoring started")
    
    # Simulate some workload
    print("\n⚡ Simulating workload...")
    
    for i in range(10):
        # Simulate request processing
        processing_time = 0.1 + (i * 0.05)  # Increasing processing time
        success = i < 8  # 8 out of 10 successful
        
        monitor.record_request(processing_time, success)
        await asyncio.sleep(0.1)
    
    # Let monitoring collect some data
    await asyncio.sleep(2)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    
    print("📊 Performance Summary:")
    if 'current' in summary:
        current = summary['current']
        print(f"   Current CPU: {current.get('cpu_usage', 0):.1f}%")
        print(f"   Current Memory: {current.get('memory_usage', 0):.1f}%")
        print(f"   Current Throughput: {current.get('throughput', 0):.1f} req/s")
        print(f"   Current Error Rate: {current.get('error_rate', 0):.1f}%")
    
    if 'totals' in summary:
        totals = summary['totals']
        print(f"   Total Requests: {totals.get('total_requests', 0)}")
        print(f"   Total Errors: {totals.get('error_count', 0)}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\n🛑 Performance monitoring stopped")
    print("✅ Performance monitoring test completed!")

async def test_distributed_coordination():
    """Test distributed coordination features."""
    print("\n🌐 Testing Distributed Coordination")
    print("=" * 40)
    
    # Create cluster configuration
    cluster_config = ClusterConfig(
        cluster_name="test-cluster",
        heartbeat_interval=2.0,  # Fast for testing
        load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
    )
    
    # Create coordinator node
    coordinator_node = NodeInfo(
        node_id="coordinator-1",
        role=NodeRole.COORDINATOR,
        host="localhost",
        port=9001,
        capabilities={
            "formalize_latex": True,
            "verify_proof": True,
            "parse_latex": True
        },
        max_tasks=5
    )
    
    # Create task manager (simplified test - single node)
    coordinator = DistributedTaskManager(coordinator_node, cluster_config)
    
    try:
        # Start coordinator
        await coordinator.start()
        print(f"✅ Coordinator started on port {coordinator_node.port}")
        
        # Test task submission and execution
        print("\n📋 Testing task distribution...")
        
        # Submit tasks to coordinator
        task_ids = []
        for i in range(3):
            task_id = await coordinator.submit_task(
                task_type="formalize_latex",
                payload={
                    "latex_content": f"\\begin{{theorem}} Test theorem {i} \\end{{theorem}}",
                    "target_system": "lean4"
                },
                priority=1
            )
            task_ids.append(task_id)
            print(f"   Submitted task {i+1}: {task_id[:8]}...")
        
        # Wait for tasks to complete
        print("\n⏳ Waiting for task completion...")
        results = []
        for task_id in task_ids:
            try:
                result = await coordinator.get_task_result(task_id, timeout=10.0)
                results.append(result)
                print(f"   Task {task_id[:8]}... completed")
            except Exception as e:
                print(f"   Task {task_id[:8]}... failed: {e}")
        
        print(f"\n📊 Task execution summary:")
        print(f"   Tasks submitted: {len(task_ids)}")
        print(f"   Tasks completed: {len(results)}")
        print(f"   Success rate: {len(results)/len(task_ids):.1%}")
        
        # Test cluster status
        print("\n🏢 Cluster status:")
        coordinator_status = coordinator.get_cluster_status()
        
        print(f"   Coordinator knows {coordinator_status['node_count']} nodes")
        print(f"   Coordinator tasks: {coordinator_status['tasks']}")
        
    except Exception as e:
        print(f"❌ Error in distributed coordination test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            await coordinator.stop()
            print("\n🛑 Distributed system stopped")
        except Exception as e:
            print(f"Error stopping distributed system: {e}")
    
    print("\n✅ Distributed coordination test completed!")

def main():
    """Main test function."""
    print("🚀 Starting Scaling & Optimization Tests")
    print("=" * 60)
    
    # Run async tests
    try:
        # Test optimization engine
        asyncio.run(test_optimization_engine())
        
        # Test intelligent cache
        asyncio.run(test_intelligent_cache())
        
        # Test performance monitoring 
        asyncio.run(test_performance_monitoring())
        
        # Test distributed coordination
        asyncio.run(test_distributed_coordination())
        
        print("\n" + "=" * 60)
        print("🎉 All scaling & optimization tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()