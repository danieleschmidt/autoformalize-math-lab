#!/usr/bin/env python3
"""Generation 6 Resilient Architecture Demo.

Demonstrates advanced resilience patterns including circuit breakers, bulkheads,
adaptive retry strategies, health monitoring, and self-healing capabilities.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import random
import sys
sys.path.append('src')

from src.autoformalize.core.generation6_resilient_architecture import (
    ResilientFormalizationPipeline,
    create_resilient_pipeline,
    CircuitBreaker,
    AdaptiveRetryStrategy,
    BulkheadIsolation,
    HealthMonitor,
    SelfHealingSystem
)


class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")


class Generation6ResilienceDemo:
    """Comprehensive demo of Generation 6 resilient architecture."""
    
    def __init__(self):
        self.logger = MockLogger()
        self.results = {
            'circuit_breaker_tests': [],
            'retry_strategy_tests': [],
            'bulkhead_tests': [],
            'health_monitoring_tests': [],
            'self_healing_tests': [],
            'integrated_resilience_tests': [],
            'performance_metrics': {}
        }
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive Generation 6 resilience demo."""
        self.logger.info("🛡️ Starting Generation 6 Resilient Architecture Demo")
        
        # Test individual resilience components
        await self._test_circuit_breakers()
        await self._test_retry_strategies()
        await self._test_bulkhead_isolation()
        await self._test_health_monitoring()
        await self._test_self_healing()
        
        # Test integrated resilient pipeline
        await self._test_integrated_resilience()
        
        self.logger.info("✅ Generation 6 Resilience Demo completed successfully")
        return self.results
    
    async def _test_circuit_breakers(self) -> None:
        """Test circuit breaker functionality."""
        self.logger.info("Testing circuit breaker patterns...")
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5,
            success_threshold=2,
            name="test_breaker"
        )
        
        # Test normal operation (closed state)
        async def successful_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        async def failing_operation():
            await asyncio.sleep(0.1)
            raise Exception("Simulated failure")
        
        test_results = {
            'test_name': 'Circuit Breaker Functionality',
            'operations': []
        }
        
        # Test successful operations
        for i in range(5):
            try:
                result = await circuit_breaker.call(successful_operation)
                test_results['operations'].append({
                    'operation_id': i + 1,
                    'type': 'success',
                    'result': 'success',
                    'state': circuit_breaker.state.value,
                    'consecutive_successes': circuit_breaker.metrics.consecutive_successes
                })
            except Exception as e:
                test_results['operations'].append({
                    'operation_id': i + 1,
                    'type': 'success',
                    'error': str(e),
                    'state': circuit_breaker.state.value
                })
        
        # Test failing operations to trigger circuit breaker
        for i in range(5):
            try:
                result = await circuit_breaker.call(failing_operation)
                test_results['operations'].append({
                    'operation_id': i + 6,
                    'type': 'failure',
                    'result': result,
                    'state': circuit_breaker.state.value
                })
            except Exception as e:
                test_results['operations'].append({
                    'operation_id': i + 6,
                    'type': 'failure',
                    'error': str(e),
                    'state': circuit_breaker.state.value,
                    'consecutive_failures': circuit_breaker.metrics.consecutive_failures
                })
        
        # Wait for recovery timeout and test half-open state
        self.logger.info("Waiting for circuit breaker recovery...")
        await asyncio.sleep(6)
        
        try:
            result = await circuit_breaker.call(successful_operation)
            test_results['operations'].append({
                'operation_id': 11,
                'type': 'recovery_test',
                'result': result,
                'state': circuit_breaker.state.value
            })
        except Exception as e:
            test_results['operations'].append({
                'operation_id': 11,
                'type': 'recovery_test',
                'error': str(e),
                'state': circuit_breaker.state.value
            })
        
        test_results['final_status'] = circuit_breaker.get_status()
        self.results['circuit_breaker_tests'].append(test_results)
        
        self.logger.info(
            f"Circuit breaker test completed. Final state: {circuit_breaker.state.value}, "
            f"Success rate: {circuit_breaker.metrics.success_rate:.3f}"
        )
    
    async def _test_retry_strategies(self) -> None:
        """Test adaptive retry strategy functionality."""
        self.logger.info("Testing adaptive retry strategies...")
        
        retry_strategy = AdaptiveRetryStrategy(
            max_retries=3,
            base_delay=0.5,
            max_delay=5.0,
            exponential_base=2.0,
            jitter_factor=0.1
        )
        
        # Test operations with different failure patterns
        test_cases = [
            {'name': 'always_succeed', 'failure_rate': 0.0},
            {'name': 'occasional_failure', 'failure_rate': 0.3},
            {'name': 'frequent_failure', 'failure_rate': 0.7},
            {'name': 'always_fail', 'failure_rate': 1.0}
        ]
        
        for case in test_cases:
            async def test_operation():
                if random.random() < case['failure_rate']:
                    raise Exception(f"Simulated failure for {case['name']}")
                return f"Success for {case['name']}"
            
            test_result = {
                'test_case': case['name'],
                'failure_rate': case['failure_rate'],
                'attempts': []
            }
            
            start_time = time.time()
            try:
                result = await retry_strategy.execute_with_retry(test_operation)
                test_result['final_result'] = result
                test_result['success'] = True
            except Exception as e:
                test_result['final_result'] = str(e)
                test_result['success'] = False
            
            test_result['total_time'] = time.time() - start_time
            test_result['recent_success_rate'] = retry_strategy._get_recent_success_rate()
            
            self.results['retry_strategy_tests'].append(test_result)
            
            self.logger.info(
                f"Retry test '{case['name']}': "
                f"Success={test_result['success']}, "
                f"Time={test_result['total_time']:.2f}s"
            )
    
    async def _test_bulkhead_isolation(self) -> None:
        """Test bulkhead isolation functionality."""
        self.logger.info("Testing bulkhead isolation...")
        
        bulkhead = BulkheadIsolation({
            'fast_operations': 3,
            'slow_operations': 2,
            'heavy_operations': 1
        })
        
        async def fast_operation(operation_id: int):
            await asyncio.sleep(0.1)
            return f"Fast operation {operation_id} completed"
        
        async def slow_operation(operation_id: int):
            await asyncio.sleep(1.0)
            return f"Slow operation {operation_id} completed"
        
        async def heavy_operation(operation_id: int):
            await asyncio.sleep(2.0)
            return f"Heavy operation {operation_id} completed"
        
        # Start multiple operations concurrently to test bulkhead limits
        tasks = []
        
        # Fast operations (should complete quickly)
        for i in range(5):
            task = asyncio.create_task(
                bulkhead.execute_in_bulkhead('fast_operations', fast_operation, i)
            )
            tasks.append(('fast', i, task))
        
        # Slow operations (should be limited)
        for i in range(3):
            task = asyncio.create_task(
                bulkhead.execute_in_bulkhead('slow_operations', slow_operation, i)
            )
            tasks.append(('slow', i, task))
        
        # Heavy operations (should be severely limited)
        for i in range(2):
            task = asyncio.create_task(
                bulkhead.execute_in_bulkhead('heavy_operations', heavy_operation, i)
            )
            tasks.append(('heavy', i, task))
        
        # Monitor bulkhead status during execution
        bulkhead_snapshots = []
        for _ in range(5):
            await asyncio.sleep(0.5)
            status = bulkhead.get_bulkhead_status()
            bulkhead_snapshots.append({
                'timestamp': time.time(),
                'status': status
            })
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
        
        test_results = {
            'operations_started': len(tasks),
            'operations_completed': len([r for r in results if not isinstance(r, Exception)]),
            'operations_failed': len([r for r in results if isinstance(r, Exception)]),
            'bulkhead_snapshots': bulkhead_snapshots,
            'final_status': bulkhead.get_bulkhead_status()
        }
        
        self.results['bulkhead_tests'].append(test_results)
        
        self.logger.info(
            f"Bulkhead test completed: "
            f"{test_results['operations_completed']}/{test_results['operations_started']} succeeded"
        )
    
    async def _test_health_monitoring(self) -> None:
        """Test health monitoring system."""
        self.logger.info("Testing health monitoring...")
        
        health_monitor = HealthMonitor(check_interval=2.0)
        
        # Create mock components to monitor
        class MockComponent:
            def __init__(self, name: str, failure_rate: float = 0.0):
                self.name = name
                self.failure_rate = failure_rate
                self.request_count = 0
                self.error_count = 0
            
            def get_status(self):
                self.request_count += 1
                if random.random() < self.failure_rate:
                    self.error_count += 1
                
                success_rate = (
                    (self.request_count - self.error_count) / self.request_count
                    if self.request_count > 0 else 1.0
                )
                
                return {
                    'metrics': {
                        'success_rate': success_rate,
                        'average_response_time': random.uniform(0.1, 2.0),
                        'error_count': self.error_count,
                        'total_requests': self.request_count
                    }
                }
        
        # Register components with different health profiles
        healthy_component = MockComponent("healthy_service", failure_rate=0.05)
        warning_component = MockComponent("warning_service", failure_rate=0.15)
        unhealthy_component = MockComponent("unhealthy_service", failure_rate=0.6)
        
        health_monitor.register_component("healthy", healthy_component)
        health_monitor.register_component("warning", warning_component)
        health_monitor.register_component("unhealthy", unhealthy_component)
        
        # Run health monitoring for a short period
        monitoring_task = asyncio.create_task(health_monitor.start_monitoring())
        
        # Collect health data
        health_snapshots = []
        for i in range(3):
            await asyncio.sleep(2.5)  # Let health checks run
            overall_health = health_monitor.get_overall_health()
            health_snapshots.append({
                'check_iteration': i + 1,
                'overall_health': overall_health,
                'timestamp': time.time()
            })
        
        # Stop monitoring
        health_monitor.stop_monitoring()
        
        # Wait a bit for the monitoring task to finish
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()
        
        test_results = {
            'components_monitored': 3,
            'health_snapshots': health_snapshots,
            'final_overall_health': health_monitor.get_overall_health(),
            'alert_count': len(health_monitor.alerts)
        }
        
        self.results['health_monitoring_tests'].append(test_results)
        
        self.logger.info(
            f"Health monitoring test completed. "
            f"Final status: {test_results['final_overall_health']['overall_status']}, "
            f"Alerts generated: {test_results['alert_count']}"
        )
    
    async def _test_self_healing(self) -> None:
        """Test self-healing system functionality."""
        self.logger.info("Testing self-healing system...")
        
        self_healing = SelfHealingSystem()
        
        # Register healing strategies
        async def heal_connection_error(error: Exception, context: Dict[str, Any]) -> bool:
            self.logger.info("Healing connection error...")
            await asyncio.sleep(0.5)  # Simulate healing time
            return random.random() > 0.2  # 80% success rate
        
        async def heal_timeout_error(error: Exception, context: Dict[str, Any]) -> bool:
            self.logger.info("Healing timeout error...")
            await asyncio.sleep(0.3)
            return random.random() > 0.3  # 70% success rate
        
        async def heal_memory_error(error: Exception, context: Dict[str, Any]) -> bool:
            self.logger.info("Healing memory error...")
            await asyncio.sleep(0.1)
            return random.random() > 0.5  # 50% success rate
        
        self_healing.register_healing_strategy("ConnectionError", heal_connection_error, max_attempts=2)
        self_healing.register_healing_strategy("TimeoutError", heal_timeout_error, max_attempts=3)
        self_healing.register_healing_strategy("MemoryError", heal_memory_error, max_attempts=1)
        
        # Test healing attempts for different error types
        test_errors = [
            (ConnectionError("Connection failed"), "connection_test"),
            (TimeoutError("Operation timed out"), "timeout_test"),
            (MemoryError("Out of memory"), "memory_test"),
            (ValueError("Unknown error"), "unknown_error_test")  # No healing strategy
        ]
        
        healing_results = []
        
        for error, test_name in test_errors:
            context = {'test_name': test_name, 'timestamp': time.time()}
            
            healing_result = {
                'test_name': test_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'healing_attempts': []
            }
            
            # Attempt healing multiple times to test max_attempts
            for attempt in range(4):  # More than max attempts for any strategy
                try:
                    success = await self_healing.attempt_healing(error, context)
                    healing_result['healing_attempts'].append({
                        'attempt': attempt + 1,
                        'success': success
                    })
                    
                    if success:
                        break
                        
                except Exception as healing_error:
                    healing_result['healing_attempts'].append({
                        'attempt': attempt + 1,
                        'error': str(healing_error)
                    })
            
            healing_results.append(healing_result)
            
            self.logger.info(
                f"Healing test for {test_name}: "
                f"{len(healing_result['healing_attempts'])} attempts"
            )
        
        test_results = {
            'healing_tests': healing_results,
            'healing_statistics': self_healing.get_healing_statistics()
        }
        
        self.results['self_healing_tests'].append(test_results)
        
        self.logger.info("Self-healing test completed")
    
    async def _test_integrated_resilience(self) -> None:
        """Test integrated resilient formalization pipeline."""
        self.logger.info("Testing integrated resilient pipeline...")
        
        pipeline = create_resilient_pipeline()
        
        # Test resilient formalization with various scenarios
        test_scenarios = [
            "Simple theorem: For any natural number n, n + 0 = n",
            "Complex theorem: The fundamental theorem of arithmetic",
            "Analysis theorem: Every continuous function on a compact set is uniformly continuous",
            "Algebra theorem: Every finite group has a unique decomposition",
            "Topology theorem: Every compact metric space is complete"
        ]
        
        formalization_results = []
        
        for i, theorem in enumerate(test_scenarios):
            self.logger.info(f"Testing resilient formalization {i+1}/{len(test_scenarios)}")
            
            start_time = time.time()
            try:
                result = await pipeline.resilient_formalize(theorem, target_system="lean4")
                
                formalization_result = {
                    'test_id': i + 1,
                    'theorem': theorem[:50] + "...",
                    'success': result['success'],
                    'processing_time': time.time() - start_time,
                    'resilience_features_used': result.get('resilience_stats', {})
                }
                
                if result['success']:
                    formalization_result['formal_code_length'] = len(result.get('formal_code', ''))
                
            except Exception as e:
                formalization_result = {
                    'test_id': i + 1,
                    'theorem': theorem[:50] + "...",
                    'success': False,
                    'processing_time': time.time() - start_time,
                    'error': str(e)
                }
            
            formalization_results.append(formalization_result)
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Get comprehensive resilience status
        resilience_status = pipeline.get_resilience_status()
        
        test_results = {
            'formalization_tests': formalization_results,
            'success_rate': sum(1 for r in formalization_results if r['success']) / len(formalization_results),
            'average_processing_time': sum(r['processing_time'] for r in formalization_results) / len(formalization_results),
            'resilience_system_status': resilience_status
        }
        
        self.results['integrated_resilience_tests'].append(test_results)
        
        self.logger.info(
            f"Integrated resilience test completed. "
            f"Success rate: {test_results['success_rate']:.3f}, "
            f"Avg time: {test_results['average_processing_time']:.2f}s"
        )


async def main():
    """Main execution function for Generation 6 Resilience Demo."""
    demo = Generation6ResilienceDemo()
    
    try:
        results = await demo.run_comprehensive_demo()
        
        # Save results
        results_path = Path("generation6_resilience_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("🛡️ Generation 6 Resilient Architecture Demo Results:")
        print("=" * 70)
        
        # Display circuit breaker results
        if results['circuit_breaker_tests']:
            cb_test = results['circuit_breaker_tests'][0]
            operations = cb_test['operations']
            successful_ops = len([op for op in operations if 'result' in op and op['result'] == 'success'])
            print(f"Circuit Breaker Test: {successful_ops}/{len(operations)} operations succeeded")
        
        # Display retry strategy results  
        if results['retry_strategy_tests']:
            retry_tests = results['retry_strategy_tests']
            avg_success_rate = sum(1 for test in retry_tests if test['success']) / len(retry_tests)
            print(f"Retry Strategy Success Rate: {avg_success_rate:.3f}")
        
        # Display bulkhead results
        if results['bulkhead_tests']:
            bulkhead_test = results['bulkhead_tests'][0]
            completed = bulkhead_test['operations_completed']
            total = bulkhead_test['operations_started']
            print(f"Bulkhead Isolation: {completed}/{total} operations completed")
        
        # Display health monitoring results
        if results['health_monitoring_tests']:
            health_test = results['health_monitoring_tests'][0]
            final_status = health_test['final_overall_health']['overall_status']
            alert_count = health_test['alert_count']
            print(f"Health Monitoring: Status={final_status}, Alerts={alert_count}")
        
        # Display self-healing results
        if results['self_healing_tests']:
            healing_test = results['self_healing_tests'][0]
            healing_stats = healing_test['healing_statistics']
            strategies = healing_stats['registered_strategies']
            print(f"Self-Healing System: {strategies} strategies registered")
        
        # Display integrated resilience results
        if results['integrated_resilience_tests']:
            integrated_test = results['integrated_resilience_tests'][0]
            success_rate = integrated_test['success_rate']
            avg_time = integrated_test['average_processing_time']
            print(f"Integrated Pipeline: Success={success_rate:.3f}, Avg Time={avg_time:.2f}s")
        
        print("=" * 70)
        print(f"Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Generation 6 Resilience Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())