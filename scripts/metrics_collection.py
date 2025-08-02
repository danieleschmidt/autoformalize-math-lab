#!/usr/bin/env python3
"""
Automated metrics collection script for autoformalize-math-lab.

This script collects various metrics from different sources and aggregates them
for monitoring, reporting, and decision making.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse
import requests
import subprocess
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects metrics from various sources."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize the metrics collector."""
        self.config_path = config_path
        self.config = self._load_config()
        self.metrics = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            sys.exit(1)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        logger.info("Starting metrics collection...")
        
        # Collect code metrics
        logger.info("Collecting code metrics...")
        self.metrics['code_metrics'] = self._collect_code_metrics()
        
        # Collect performance metrics
        logger.info("Collecting performance metrics...")
        self.metrics['performance_metrics'] = self._collect_performance_metrics()
        
        # Collect business metrics
        logger.info("Collecting business metrics...")
        self.metrics['business_metrics'] = self._collect_business_metrics()
        
        # Collect system metrics
        logger.info("Collecting system metrics...")
        self.metrics['system_metrics'] = self._collect_system_metrics()
        
        # Add timestamp and metadata
        self.metrics['collection_timestamp'] = datetime.utcnow().isoformat()
        self.metrics['collection_version'] = "1.0.0"
        
        logger.info("Metrics collection completed")
        return self.metrics
    
    def _collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code quality and complexity metrics."""
        metrics = {}
        
        try:
            # Lines of code
            metrics['lines_of_code'] = self._count_lines_of_code()
            
            # Test coverage
            metrics['test_coverage'] = self._get_test_coverage()
            
            # Cyclomatic complexity
            metrics['cyclomatic_complexity'] = self._calculate_complexity()
            
            # Technical debt
            metrics['technical_debt'] = self._assess_technical_debt()
            
            # Security vulnerabilities
            metrics['security_vulnerabilities'] = self._scan_vulnerabilities()
            
        except Exception as e:
            logger.error(f"Error collecting code metrics: {e}")
            
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {}
        
        try:
            # Query Prometheus for performance metrics
            prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
            
            # Formalization success rate
            metrics['formalization_success_rate'] = self._query_prometheus(
                prometheus_url, 
                'formalization_success_rate'
            )
            
            # Average response time
            metrics['avg_response_time'] = self._query_prometheus(
                prometheus_url,
                'histogram_quantile(0.95, rate(formalization_duration_seconds_bucket[5m]))'
            )
            
            # Error rate
            metrics['error_rate'] = self._query_prometheus(
                prometheus_url,
                'rate(formalizations_total{status="error"}[5m])'
            )
            
            # LLM token usage
            metrics['llm_token_usage'] = self._query_prometheus(
                prometheus_url,
                'rate(llm_tokens_used_total[1h])'
            )
            
            # Active formalizations
            metrics['active_formalizations'] = self._query_prometheus(
                prometheus_url,
                'active_formalizations'
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            
        return metrics
    
    def _collect_business_metrics(self) -> Dict[str, Any]:
        """Collect business and usage metrics."""
        metrics = {}
        
        try:
            # Daily formalization count
            metrics['daily_formalizations'] = self._count_daily_formalizations()
            
            # Unique users
            metrics['unique_users'] = self._count_unique_users()
            
            # API usage
            metrics['api_calls'] = self._count_api_calls()
            
            # Cost metrics
            metrics['daily_cost'] = self._calculate_daily_cost()
            
            # Success by domain
            metrics['success_by_domain'] = self._analyze_success_by_domain()
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
            
        return metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        metrics = {}
        
        try:
            # CPU and memory usage
            import psutil
            
            metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
            metrics['memory_usage_percent'] = psutil.virtual_memory().percent
            metrics['disk_usage_percent'] = psutil.disk_usage('/').percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics['network_bytes_sent'] = net_io.bytes_sent
            metrics['network_bytes_recv'] = net_io.bytes_recv
            
            # Process information
            metrics['active_processes'] = len(psutil.pids())
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
    
    def _count_lines_of_code(self) -> Dict[str, int]:
        """Count lines of code in the project."""
        try:
            result = subprocess.run([
                'find', 'src/', '-name', '*.py', '-exec', 'wc', '-l', '{}', '+'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_lines = int(lines[-1].strip().split()[0])
                return {
                    'total_lines': total_lines,
                    'source_files_count': len(lines) - 1
                }
        except Exception as e:
            logger.error(f"Error counting lines of code: {e}")
            
        return {'total_lines': 0, 'source_files_count': 0}
    
    def _get_test_coverage(self) -> Dict[str, float]:
        """Get test coverage information."""
        try:
            # Run coverage report
            result = subprocess.run([
                'python', '-m', 'pytest', '--cov=src/autoformalize', 
                '--cov-report=json', 'tests/'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                    return {
                        'line_coverage': coverage_data['totals']['percent_covered'],
                        'branch_coverage': coverage_data['totals'].get('percent_covered_display', 0),
                        'total_lines': coverage_data['totals']['num_statements'],
                        'covered_lines': coverage_data['totals']['covered_lines']
                    }
        except Exception as e:
            logger.error(f"Error getting test coverage: {e}")
            
        return {'line_coverage': 0, 'branch_coverage': 0}
    
    def _calculate_complexity(self) -> Dict[str, float]:
        """Calculate cyclomatic complexity."""
        try:
            result = subprocess.run([
                'python', '-m', 'radon', 'cc', 'src/', '--json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            total_complexity += item['complexity']
                            function_count += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                return {
                    'average_complexity': avg_complexity,
                    'total_complexity': total_complexity,
                    'function_count': function_count
                }
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            
        return {'average_complexity': 0, 'total_complexity': 0, 'function_count': 0}
    
    def _assess_technical_debt(self) -> Dict[str, Any]:
        """Assess technical debt in the codebase."""
        try:
            # Run code quality analysis
            result = subprocess.run([
                'python', '-m', 'pylint', 'src/', '--output-format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                issue_counts = {}
                for issue in issues:
                    issue_type = issue['type']
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
                return {
                    'total_issues': len(issues),
                    'issue_breakdown': issue_counts,
                    'technical_debt_ratio': len(issues) / self._count_lines_of_code()['total_lines'] * 100
                }
        except Exception as e:
            logger.error(f"Error assessing technical debt: {e}")
            
        return {'total_issues': 0, 'issue_breakdown': {}, 'technical_debt_ratio': 0}
    
    def _scan_vulnerabilities(self) -> Dict[str, Any]:
        """Scan for security vulnerabilities."""
        try:
            # Run safety check
            result = subprocess.run([
                'python', '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            vulnerabilities = []
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = safety_data
                except json.JSONDecodeError:
                    pass
            
            return {
                'vulnerability_count': len(vulnerabilities),
                'vulnerabilities': vulnerabilities[:10]  # Limit to first 10
            }
        except Exception as e:
            logger.error(f"Error scanning vulnerabilities: {e}")
            
        return {'vulnerability_count': 0, 'vulnerabilities': []}
    
    def _query_prometheus(self, base_url: str, query: str) -> Optional[float]:
        """Query Prometheus for metrics."""
        try:
            response = requests.get(
                f"{base_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    return float(data['data']['result'][0]['value'][1])
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            
        return None
    
    def _count_daily_formalizations(self) -> int:
        """Count formalizations in the last 24 hours."""
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        result = self._query_prometheus(
            prometheus_url,
            'increase(formalizations_total[24h])'
        )
        return int(result) if result else 0
    
    def _count_unique_users(self) -> int:
        """Count unique users in the last 24 hours."""
        # This would typically query your analytics system
        # For now, return a placeholder
        return 42
    
    def _count_api_calls(self) -> int:
        """Count API calls in the last 24 hours."""
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        result = self._query_prometheus(
            prometheus_url,
            'increase(http_requests_total[24h])'
        )
        return int(result) if result else 0
    
    def _calculate_daily_cost(self) -> float:
        """Calculate daily operational cost."""
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        result = self._query_prometheus(
            prometheus_url,
            'increase(llm_cost_usd_total[24h])'
        )
        return float(result) if result else 0.0
    
    def _analyze_success_by_domain(self) -> Dict[str, float]:
        """Analyze success rates by mathematical domain."""
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        
        domains = ['algebra', 'analysis', 'topology', 'number_theory']
        success_rates = {}
        
        for domain in domains:
            success_rate = self._query_prometheus(
                prometheus_url,
                f'formalization_success_rate{{domain="{domain}"}}'
            )
            success_rates[domain] = success_rate if success_rate else 0.0
            
        return success_rates

class MetricsReporter:
    """Generates reports from collected metrics."""
    
    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
    
    def generate_summary_report(self) -> str:
        """Generate a summary report."""
        report = []
        report.append("# Autoformalize Metrics Summary")
        report.append(f"Generated: {self.metrics.get('collection_timestamp', 'Unknown')}")
        report.append("")
        
        # Code Quality
        if 'code_metrics' in self.metrics:
            code = self.metrics['code_metrics']
            report.append("## Code Quality")
            report.append(f"- Lines of Code: {code.get('lines_of_code', {}).get('total_lines', 'N/A')}")
            report.append(f"- Test Coverage: {code.get('test_coverage', {}).get('line_coverage', 'N/A')}%")
            report.append(f"- Average Complexity: {code.get('cyclomatic_complexity', {}).get('average_complexity', 'N/A')}")
            report.append(f"- Security Issues: {code.get('security_vulnerabilities', {}).get('vulnerability_count', 'N/A')}")
            report.append("")
        
        # Performance
        if 'performance_metrics' in self.metrics:
            perf = self.metrics['performance_metrics']
            report.append("## Performance")
            report.append(f"- Success Rate: {perf.get('formalization_success_rate', 'N/A')}%")
            report.append(f"- Avg Response Time: {perf.get('avg_response_time', 'N/A')}s")
            report.append(f"- Error Rate: {perf.get('error_rate', 'N/A')}%")
            report.append(f"- Active Processes: {perf.get('active_formalizations', 'N/A')}")
            report.append("")
        
        # Business Metrics
        if 'business_metrics' in self.metrics:
            business = self.metrics['business_metrics']
            report.append("## Business Metrics")
            report.append(f"- Daily Formalizations: {business.get('daily_formalizations', 'N/A')}")
            report.append(f"- Unique Users: {business.get('unique_users', 'N/A')}")
            report.append(f"- Daily Cost: ${business.get('daily_cost', 'N/A')}")
            report.append("")
        
        return "\n".join(report)
    
    def export_json(self, filename: str):
        """Export metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def export_csv(self, filename: str):
        """Export metrics to CSV file."""
        import csv
        
        # Flatten metrics for CSV export
        flattened = self._flatten_dict(self.metrics)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in flattened.items():
                writer.writerow([key, value])
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Collect and report project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                        help='Path to metrics configuration file')
    parser.add_argument('--output-json', help='Output JSON file path')
    parser.add_argument('--output-csv', help='Output CSV file path')
    parser.add_argument('--output-report', help='Output report file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Collect metrics
        collector = MetricsCollector(args.config)
        metrics = collector.collect_all_metrics()
        
        # Generate reports
        reporter = MetricsReporter(metrics)
        
        if args.output_json:
            reporter.export_json(args.output_json)
            logger.info(f"Metrics exported to {args.output_json}")
        
        if args.output_csv:
            reporter.export_csv(args.output_csv)
            logger.info(f"Metrics exported to {args.output_csv}")
        
        if args.output_report:
            with open(args.output_report, 'w') as f:
                f.write(reporter.generate_summary_report())
            logger.info(f"Report generated: {args.output_report}")
        
        # Print summary to console
        print(reporter.generate_summary_report())
        
    except Exception as e:
        logger.error(f"Error in metrics collection: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()