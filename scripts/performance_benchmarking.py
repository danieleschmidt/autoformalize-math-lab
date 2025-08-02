#!/usr/bin/env python3
"""
Performance benchmarking script for autoformalize-math-lab.

This script runs comprehensive performance benchmarks and generates reports
for tracking performance trends and detecting regressions.
"""

import json
import logging
import os
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import argparse
import subprocess
import tempfile
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Runs performance benchmarks."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarks...")
        
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
        # Core performance benchmarks
        self.results['benchmarks']['parsing'] = self._benchmark_parsing()
        self.results['benchmarks']['generation'] = self._benchmark_generation()
        self.results['benchmarks']['verification'] = self._benchmark_verification()
        self.results['benchmarks']['end_to_end'] = self._benchmark_end_to_end()
        
        # System performance benchmarks
        self.results['benchmarks']['memory'] = self._benchmark_memory_usage()
        self.results['benchmarks']['cpu'] = self._benchmark_cpu_usage()
        self.results['benchmarks']['io'] = self._benchmark_io_performance()
        
        # Integration benchmarks
        self.results['benchmarks']['llm_api'] = self._benchmark_llm_api()
        self.results['benchmarks']['concurrent'] = self._benchmark_concurrent_processing()
        
        logger.info("All benchmarks completed")
        return self.results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
        }
    
    def _benchmark_parsing(self) -> Dict[str, Any]:
        """Benchmark LaTeX parsing performance."""
        logger.info("Running parsing benchmarks...")
        
        # Sample LaTeX inputs of varying complexity
        test_cases = [
            {
                'name': 'simple_theorem',
                'content': r'\begin{theorem}For any $n \in \mathbb{N}$, $n + 0 = n$.\end{theorem}',
                'iterations': 100
            },
            {
                'name': 'complex_proof',
                'content': r'''
                \begin{theorem}
                For any prime $p > 2$, we have $p \equiv 1 \pmod{4}$ or $p \equiv 3 \pmod{4}$.
                \end{theorem}
                \begin{proof}
                Since $p$ is odd and greater than 2, $p$ is not divisible by 2.
                By the division algorithm, $p = 4q + r$ where $r \in \{0, 1, 2, 3\}$.
                Since $p$ is odd, $r \neq 0$ and $r \neq 2$, thus $r \in \{1, 3\}$.
                \end{proof}
                ''',
                'iterations': 50
            },
            {
                'name': 'large_document',
                'content': r'\section{Introduction}' + r'\begin{theorem}Test theorem.\end{theorem}' * 20,
                'iterations': 10
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            times = []
            memory_usage = []
            
            for i in range(test_case['iterations']):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    # Simulate parsing (replace with actual parsing call)
                    self._simulate_parsing(test_case['content'])
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    times.append(end_time - start_time)
                    memory_usage.append(end_memory - start_memory)
                    
                except Exception as e:
                    logger.error(f"Error in parsing benchmark {test_case['name']}: {e}")
                    continue
            
            if times:
                results[test_case['name']] = {
                    'avg_time_ms': statistics.mean(times) * 1000,
                    'median_time_ms': statistics.median(times) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'std_dev_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                    'avg_memory_mb': statistics.mean(memory_usage),
                    'iterations': len(times)
                }
        
        return results
    
    def _benchmark_generation(self) -> Dict[str, Any]:
        """Benchmark formal proof generation performance."""
        logger.info("Running generation benchmarks...")
        
        test_cases = [
            {
                'name': 'basic_arithmetic',
                'input': 'Prove that 2 + 2 = 4',
                'target': 'lean4',
                'iterations': 20
            },
            {
                'name': 'algebraic_identity',
                'input': 'Prove that (a + b)² = a² + 2ab + b²',
                'target': 'lean4',
                'iterations': 10
            },
            {
                'name': 'number_theory',
                'input': 'Prove that there are infinitely many prime numbers',
                'target': 'lean4',
                'iterations': 5
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            times = []
            success_count = 0
            
            for i in range(test_case['iterations']):
                start_time = time.time()
                
                try:
                    # Simulate generation (replace with actual generation call)
                    success = self._simulate_generation(
                        test_case['input'], 
                        test_case['target']
                    )
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error in generation benchmark {test_case['name']}: {e}")
                    continue
            
            if times:
                results[test_case['name']] = {
                    'avg_time_ms': statistics.mean(times) * 1000,
                    'median_time_ms': statistics.median(times) * 1000,
                    'success_rate': success_count / len(times),
                    'iterations': len(times)
                }
        
        return results
    
    def _benchmark_verification(self) -> Dict[str, Any]:
        """Benchmark proof verification performance."""
        logger.info("Running verification benchmarks...")
        
        # Sample Lean proofs for verification
        test_proofs = [
            {
                'name': 'simple_tactic',
                'proof': 'theorem test : True := trivial',
                'iterations': 50
            },
            {
                'name': 'arithmetic_proof',
                'proof': '''
                theorem add_zero (n : ℕ) : n + 0 = n := by
                  rw [Nat.add_zero]
                ''',
                'iterations': 20
            },
            {
                'name': 'complex_proof',
                'proof': '''
                theorem example_proof (a b : ℕ) : a + b = b + a := by
                  rw [Nat.add_comm]
                ''',
                'iterations': 10
            }
        ]
        
        results = {}
        
        for test_proof in test_proofs:
            times = []
            verification_success = 0
            
            for i in range(test_proof['iterations']):
                start_time = time.time()
                
                try:
                    # Simulate verification (replace with actual verification call)
                    success = self._simulate_verification(test_proof['proof'])
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                    if success:
                        verification_success += 1
                        
                except Exception as e:
                    logger.error(f"Error in verification benchmark {test_proof['name']}: {e}")
                    continue
            
            if times:
                results[test_proof['name']] = {
                    'avg_time_ms': statistics.mean(times) * 1000,
                    'median_time_ms': statistics.median(times) * 1000,
                    'verification_success_rate': verification_success / len(times),
                    'iterations': len(times)
                }
        
        return results
    
    def _benchmark_end_to_end(self) -> Dict[str, Any]:
        """Benchmark complete end-to-end pipeline."""
        logger.info("Running end-to-end benchmarks...")
        
        test_scenarios = [
            {
                'name': 'simple_formalization',
                'latex_input': r'\begin{theorem}$1 + 1 = 2$\end{theorem}',
                'target_system': 'lean4',
                'iterations': 5
            },
            {
                'name': 'theorem_with_proof',
                'latex_input': r'''
                \begin{theorem}
                For any natural number $n$, $n + 0 = n$.
                \end{theorem}
                \begin{proof}
                By the definition of addition.
                \end{proof}
                ''',
                'target_system': 'lean4',
                'iterations': 3
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            times = []
            success_count = 0
            
            for i in range(scenario['iterations']):
                start_time = time.time()
                
                try:
                    # Simulate end-to-end pipeline
                    success = self._simulate_end_to_end_pipeline(
                        scenario['latex_input'],
                        scenario['target_system']
                    )
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error in end-to-end benchmark {scenario['name']}: {e}")
                    continue
            
            if times:
                results[scenario['name']] = {
                    'avg_time_s': statistics.mean(times),
                    'median_time_s': statistics.median(times),
                    'success_rate': success_count / len(times),
                    'iterations': len(times)
                }
        
        return results
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("Running memory benchmarks...")
        
        import psutil
        import gc
        
        results = {}
        
        # Memory usage during different operations
        operations = [
            ('baseline', lambda: time.sleep(0.1)),
            ('large_data_processing', self._simulate_large_data_processing),
            ('multiple_formalizations', self._simulate_multiple_formalizations)
        ]
        
        for op_name, operation in operations:
            memory_samples = []
            
            # Collect memory samples during operation
            for i in range(5):
                gc.collect()  # Clean up before measurement
                start_memory = psutil.virtual_memory().used / (1024**2)  # MB
                
                operation()
                
                end_memory = psutil.virtual_memory().used / (1024**2)  # MB
                memory_samples.append(end_memory - start_memory)
            
            results[op_name] = {
                'avg_memory_increase_mb': statistics.mean(memory_samples),
                'max_memory_increase_mb': max(memory_samples),
                'min_memory_increase_mb': min(memory_samples)
            }
        
        return results
    
    def _benchmark_cpu_usage(self) -> Dict[str, Any]:
        """Benchmark CPU usage patterns."""
        logger.info("Running CPU benchmarks...")
        
        import psutil
        
        results = {}
        
        # CPU-intensive operations
        operations = [
            ('parsing_intensive', self._simulate_cpu_intensive_parsing),
            ('generation_intensive', self._simulate_cpu_intensive_generation)
        ]
        
        for op_name, operation in operations:
            cpu_samples = []
            
            for i in range(3):
                # Monitor CPU usage during operation
                psutil.cpu_percent()  # Initialize
                time.sleep(0.1)
                
                start_time = time.time()
                operation()
                duration = time.time() - start_time
                
                cpu_usage = psutil.cpu_percent()
                cpu_samples.append(cpu_usage)
            
            results[op_name] = {
                'avg_cpu_percent': statistics.mean(cpu_samples),
                'max_cpu_percent': max(cpu_samples)
            }
        
        return results
    
    def _benchmark_io_performance(self) -> Dict[str, Any]:
        """Benchmark I/O performance."""
        logger.info("Running I/O benchmarks...")
        
        results = {}
        
        # File I/O operations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write performance
            write_times = []
            for i in range(10):
                test_file = temp_path / f"test_{i}.txt"
                test_data = "test data " * 1000  # ~9KB
                
                start_time = time.time()
                with open(test_file, 'w') as f:
                    f.write(test_data)
                write_times.append(time.time() - start_time)
            
            # Read performance
            read_times = []
            for i in range(10):
                test_file = temp_path / f"test_{i}.txt"
                
                start_time = time.time()
                with open(test_file, 'r') as f:
                    content = f.read()
                read_times.append(time.time() - start_time)
        
        results['file_io'] = {
            'avg_write_time_ms': statistics.mean(write_times) * 1000,
            'avg_read_time_ms': statistics.mean(read_times) * 1000
        }
        
        return results
    
    def _benchmark_llm_api(self) -> Dict[str, Any]:
        """Benchmark LLM API performance."""
        logger.info("Running LLM API benchmarks...")
        
        # Simulate LLM API calls with different request sizes
        test_cases = [
            {'name': 'small_request', 'tokens': 100, 'iterations': 10},
            {'name': 'medium_request', 'tokens': 500, 'iterations': 5},
            {'name': 'large_request', 'tokens': 1000, 'iterations': 3}
        ]
        
        results = {}
        
        for test_case in test_cases:
            response_times = []
            
            for i in range(test_case['iterations']):
                start_time = time.time()
                
                # Simulate API call delay based on token count
                delay = test_case['tokens'] * 0.001  # 1ms per token
                time.sleep(delay)
                
                response_times.append(time.time() - start_time)
            
            results[test_case['name']] = {
                'avg_response_time_ms': statistics.mean(response_times) * 1000,
                'median_response_time_ms': statistics.median(response_times) * 1000,
                'tokens': test_case['tokens']
            }
        
        return results
    
    def _benchmark_concurrent_processing(self) -> Dict[str, Any]:
        """Benchmark concurrent processing performance."""
        logger.info("Running concurrent processing benchmarks...")
        
        import concurrent.futures
        import threading
        
        results = {}
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        
        for level in concurrency_levels:
            times = []
            
            for iteration in range(3):
                start_time = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                    futures = []
                    
                    for i in range(level * 2):  # Submit 2x tasks per worker
                        future = executor.submit(self._simulate_concurrent_task)
                        futures.append(future)
                    
                    # Wait for all tasks to complete
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                
                times.append(time.time() - start_time)
            
            results[f'concurrency_{level}'] = {
                'avg_time_s': statistics.mean(times),
                'workers': level,
                'tasks_per_worker': 2
            }
        
        return results
    
    # Simulation methods (replace with actual implementations)
    
    def _simulate_parsing(self, content: str) -> bool:
        """Simulate LaTeX parsing."""
        # Simulate parsing work
        time.sleep(len(content) * 0.00001)  # Proportional to content length
        return True
    
    def _simulate_generation(self, input_text: str, target: str) -> bool:
        """Simulate proof generation."""
        # Simulate generation work
        time.sleep(len(input_text) * 0.0001 + 0.1)  # Base time + proportional
        return True
    
    def _simulate_verification(self, proof: str) -> bool:
        """Simulate proof verification."""
        # Simulate verification work
        time.sleep(len(proof) * 0.00005 + 0.05)
        return True
    
    def _simulate_end_to_end_pipeline(self, latex_input: str, target_system: str) -> bool:
        """Simulate complete pipeline."""
        # Simulate all stages
        self._simulate_parsing(latex_input)
        self._simulate_generation(latex_input, target_system)
        self._simulate_verification("generated proof")
        return True
    
    def _simulate_large_data_processing(self):
        """Simulate processing large datasets."""
        # Create some temporary data structures
        data = list(range(100000))
        processed = [x * 2 for x in data]
        del data, processed
    
    def _simulate_multiple_formalizations(self):
        """Simulate multiple concurrent formalizations."""
        for i in range(10):
            self._simulate_parsing(f"theorem {i}")
    
    def _simulate_cpu_intensive_parsing(self):
        """Simulate CPU-intensive parsing."""
        # Simulate complex parsing work
        result = 0
        for i in range(1000000):
            result += i ** 0.5
    
    def _simulate_cpu_intensive_generation(self):
        """Simulate CPU-intensive generation."""
        # Simulate complex generation work
        data = []
        for i in range(50000):
            data.append(str(i) * 2)
        del data
    
    def _simulate_concurrent_task(self) -> bool:
        """Simulate a concurrent task."""
        time.sleep(0.1)  # Simulate some work
        return True
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        return psutil.virtual_memory().used / (1024**2)

class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates reports."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of benchmark results."""
        report = []
        report.append("# Performance Benchmark Report")
        report.append(f"Generated: {self.results.get('timestamp', 'Unknown')}")
        report.append("")
        
        # System information
        if 'system_info' in self.results:
            sys_info = self.results['system_info']
            report.append("## System Information")
            report.append(f"- Platform: {sys_info.get('platform', 'Unknown')}")
            report.append(f"- Python: {sys_info.get('python_version', 'Unknown')}")
            report.append(f"- CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
            report.append(f"- Memory: {sys_info.get('memory_total_gb', 0):.1f} GB")
            report.append("")
        
        # Benchmark results
        if 'benchmarks' in self.results:
            benchmarks = self.results['benchmarks']
            
            # Parsing performance
            if 'parsing' in benchmarks:
                report.append("## Parsing Performance")
                for test_name, metrics in benchmarks['parsing'].items():
                    report.append(f"### {test_name}")
                    report.append(f"- Average time: {metrics.get('avg_time_ms', 0):.2f} ms")
                    report.append(f"- Median time: {metrics.get('median_time_ms', 0):.2f} ms")
                    report.append(f"- Memory usage: {metrics.get('avg_memory_mb', 0):.2f} MB")
                    report.append("")
            
            # Generation performance
            if 'generation' in benchmarks:
                report.append("## Generation Performance")
                for test_name, metrics in benchmarks['generation'].items():
                    report.append(f"### {test_name}")
                    report.append(f"- Average time: {metrics.get('avg_time_ms', 0):.2f} ms")
                    report.append(f"- Success rate: {metrics.get('success_rate', 0):.2%}")
                    report.append("")
            
            # End-to-end performance
            if 'end_to_end' in benchmarks:
                report.append("## End-to-End Performance")
                for test_name, metrics in benchmarks['end_to_end'].items():
                    report.append(f"### {test_name}")
                    report.append(f"- Average time: {metrics.get('avg_time_s', 0):.2f} s")
                    report.append(f"- Success rate: {metrics.get('success_rate', 0):.2%}")
                    report.append("")
            
            # Resource usage
            if 'memory' in benchmarks:
                report.append("## Memory Usage")
                for op_name, metrics in benchmarks['memory'].items():
                    report.append(f"- {op_name}: {metrics.get('avg_memory_increase_mb', 0):.2f} MB")
                report.append("")
        
        return "\n".join(report)
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare results with baseline performance."""
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            
            comparison = {}
            
            # Compare key metrics
            current_benchmarks = self.results.get('benchmarks', {})
            baseline_benchmarks = baseline.get('benchmarks', {})
            
            for category in ['parsing', 'generation', 'end_to_end']:
                if category in current_benchmarks and category in baseline_benchmarks:
                    comparison[category] = self._compare_category(
                        current_benchmarks[category],
                        baseline_benchmarks[category]
                    )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return {}
    
    def _compare_category(self, current: Dict, baseline: Dict) -> Dict[str, Any]:
        """Compare a category of benchmarks."""
        comparison = {}
        
        for test_name in current.keys():
            if test_name in baseline:
                current_metrics = current[test_name]
                baseline_metrics = baseline[test_name]
                
                comparison[test_name] = {}
                
                # Compare time metrics
                for metric in ['avg_time_ms', 'avg_time_s']:
                    if metric in current_metrics and metric in baseline_metrics:
                        current_val = current_metrics[metric]
                        baseline_val = baseline_metrics[metric]
                        
                        if baseline_val > 0:
                            change_percent = ((current_val - baseline_val) / baseline_val) * 100
                            comparison[test_name][metric] = {
                                'current': current_val,
                                'baseline': baseline_val,
                                'change_percent': change_percent
                            }
        
        return comparison
    
    def export_results(self, filename: str):
        """Export results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--output-dir', default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--baseline', help='Baseline file for comparison')
    parser.add_argument('--export-json', help='Export results to JSON file')
    parser.add_argument('--export-report', help='Export summary report')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run benchmarks
        runner = BenchmarkRunner(args.output_dir)
        results = runner.run_all_benchmarks()
        
        # Analyze results
        analyzer = BenchmarkAnalyzer(results)
        
        # Generate and display report
        report = analyzer.generate_summary_report()
        print(report)
        
        # Export results
        if args.export_json:
            analyzer.export_results(args.export_json)
            logger.info(f"Results exported to {args.export_json}")
        
        if args.export_report:
            with open(args.export_report, 'w') as f:
                f.write(report)
            logger.info(f"Report exported to {args.export_report}")
        
        # Compare with baseline
        if args.baseline:
            comparison = analyzer.compare_with_baseline(args.baseline)
            if comparison:
                print("\n# Performance Comparison")
                for category, results in comparison.items():
                    print(f"\n## {category.title()}")
                    for test_name, metrics in results.items():
                        print(f"### {test_name}")
                        for metric_name, data in metrics.items():
                            change = data['change_percent']
                            status = "🔺" if change > 5 else "🔻" if change < -5 else "➡️"
                            print(f"- {metric_name}: {status} {change:+.1f}%")
        
        logger.info("Benchmarking completed successfully")
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()