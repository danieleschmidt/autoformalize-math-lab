#!/usr/bin/env python3
"""
Repository maintenance automation script for autoformalize-math-lab.

This script automates various repository maintenance tasks including cleanup,
optimization, and health monitoring.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RepositoryMaintainer:
    """Handles repository maintenance tasks."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.git_root = self._find_git_root()
        
    def _find_git_root(self) -> Path:
        """Find the git repository root."""
        current = self.project_root.absolute()
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return self.project_root
    
    def cleanup_temporary_files(self) -> Dict[str, int]:
        """Clean up temporary and cache files."""
        logger.info("Cleaning up temporary files...")
        
        cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo", 
            "**/*.pyd",
            "**/.pytest_cache",
            "**/.coverage",
            "**/coverage.xml",
            "**/htmlcov",
            "**/.mypy_cache",
            "**/.tox",
            "**/dist",
            "**/build",
            "**/*.egg-info",
            "**/node_modules",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.temp",
            "**/*.log"
        ]
        
        removed_files = 0
        removed_dirs = 0
        freed_space = 0
        
        for pattern in cleanup_patterns:
            for path in self.project_root.glob(pattern):
                try:
                    if path.is_file():
                        size = path.stat().st_size
                        path.unlink()
                        removed_files += 1
                        freed_space += size
                    elif path.is_dir():
                        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        shutil.rmtree(path)
                        removed_dirs += 1
                        freed_space += size
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {e}")
        
        return {
            'removed_files': removed_files,
            'removed_dirs': removed_dirs,
            'freed_space_bytes': freed_space,
            'freed_space_mb': freed_space / (1024 * 1024)
        }
    
    def optimize_git_repository(self) -> Dict[str, str]:
        """Optimize git repository."""
        logger.info("Optimizing git repository...")
        
        results = {}
        
        try:
            # Git garbage collection
            result = subprocess.run([
                'git', 'gc', '--aggressive', '--prune=now'
            ], capture_output=True, text=True, cwd=self.git_root)
            
            if result.returncode == 0:
                results['gc'] = 'success'
                logger.info("Git garbage collection completed")
            else:
                results['gc'] = f'failed: {result.stderr}'
                
        except Exception as e:
            results['gc'] = f'error: {e}'
        
        try:
            # Repack repository
            result = subprocess.run([
                'git', 'repack', '-ad'
            ], capture_output=True, text=True, cwd=self.git_root)
            
            if result.returncode == 0:
                results['repack'] = 'success'
                logger.info("Git repack completed")
            else:
                results['repack'] = f'failed: {result.stderr}'
                
        except Exception as e:
            results['repack'] = f'error: {e}'
        
        try:
            # Prune remote tracking branches
            result = subprocess.run([
                'git', 'remote', 'prune', 'origin'
            ], capture_output=True, text=True, cwd=self.git_root)
            
            if result.returncode == 0:
                results['prune_remote'] = 'success'
                logger.info("Remote branch pruning completed")
            else:
                results['prune_remote'] = f'failed: {result.stderr}'
                
        except Exception as e:
            results['prune_remote'] = f'error: {e}'
        
        return results
    
    def check_repository_health(self) -> Dict[str, any]:
        """Check repository health metrics."""
        logger.info("Checking repository health...")
        
        health = {}
        
        # Repository size
        try:
            result = subprocess.run([
                'du', '-sh', str(self.git_root)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                health['repository_size'] = result.stdout.strip().split()[0]
        except Exception as e:
            health['repository_size'] = f'error: {e}'
        
        # Git object count
        try:
            result = subprocess.run([
                'git', 'count-objects', '-v'
            ], capture_output=True, text=True, cwd=self.git_root)
            
            if result.returncode == 0:
                git_objects = {}
                for line in result.stdout.strip().split('\n'):
                    if ' ' in line:
                        key, value = line.split(' ', 1)
                        git_objects[key] = value
                health['git_objects'] = git_objects
        except Exception as e:
            health['git_objects'] = f'error: {e}'
        
        # Branch information
        try:
            result = subprocess.run([
                'git', 'branch', '-r'
            ], capture_output=True, text=True, cwd=self.git_root)
            
            if result.returncode == 0:
                remote_branches = len([b for b in result.stdout.strip().split('\n') if b.strip()])
                health['remote_branches'] = remote_branches
        except Exception as e:
            health['remote_branches'] = f'error: {e}'
        
        # Check for large files
        try:
            result = subprocess.run([
                'find', str(self.project_root), '-type', 'f', '-size', '+10M'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                large_files = [f for f in result.stdout.strip().split('\n') if f]
                health['large_files'] = large_files
                health['large_files_count'] = len(large_files)
        except Exception as e:
            health['large_files'] = f'error: {e}'
        
        # Code quality metrics
        health['code_metrics'] = self._collect_code_metrics()
        
        return health
    
    def _collect_code_metrics(self) -> Dict[str, any]:
        """Collect basic code metrics."""
        metrics = {}
        
        # Count files by type
        file_counts = {}
        for suffix in ['.py', '.md', '.yml', '.yaml', '.json', '.txt']:
            count = len(list(self.project_root.rglob(f'*{suffix}')))
            file_counts[suffix] = count
        
        metrics['file_counts'] = file_counts
        
        # Lines of code
        try:
            result = subprocess.run([
                'find', str(self.project_root / 'src'), '-name', '*.py', 
                '-exec', 'wc', '-l', '{}', '+'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    total_line = lines[-1]
                    if total_line and total_line.strip():
                        total_lines = int(total_line.strip().split()[0])
                        metrics['lines_of_code'] = total_lines
        except Exception as e:
            logger.warning(f"Could not count lines of code: {e}")
        
        return metrics
    
    def update_documentation_links(self) -> Dict[str, int]:
        """Check and update documentation links."""
        logger.info("Checking documentation links...")
        
        markdown_files = list(self.project_root.rglob('*.md'))
        
        checked_links = 0
        broken_links = 0
        updated_links = 0
        
        # This is a simplified version - in practice you'd want more sophisticated link checking
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for markdown links [text](url)
                import re
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                
                for link_text, url in links:
                    checked_links += 1
                    
                    # Check if it's a relative link to a file that exists
                    if not url.startswith(('http://', 'https://', 'mailto:')):
                        if url.startswith('/'):
                            # Absolute path from project root
                            target = self.project_root / url.lstrip('/')
                        else:
                            # Relative path from current file
                            target = md_file.parent / url
                        
                        if not target.exists():
                            logger.warning(f"Broken link in {md_file}: {url}")
                            broken_links += 1
                
            except Exception as e:
                logger.warning(f"Could not check links in {md_file}: {e}")
        
        return {
            'checked_links': checked_links,
            'broken_links': broken_links,
            'updated_links': updated_links
        }
    
    def organize_imports(self) -> Dict[str, int]:
        """Organize imports in Python files."""
        logger.info("Organizing imports...")
        
        python_files = list(self.project_root.rglob('*.py'))
        processed_files = 0
        fixed_files = 0
        
        for py_file in python_files:
            # Skip virtual environment and build directories
            if any(part in str(py_file) for part in ['venv', '.venv', 'env', 'build', 'dist']):
                continue
            
            try:
                result = subprocess.run([
                    'python', '-m', 'isort', '--check-only', str(py_file)
                ], capture_output=True, text=True)
                
                processed_files += 1
                
                if result.returncode != 0:
                    # Fix imports
                    fix_result = subprocess.run([
                        'python', '-m', 'isort', str(py_file)
                    ], capture_output=True, text=True)
                    
                    if fix_result.returncode == 0:
                        fixed_files += 1
                        logger.info(f"Fixed imports in {py_file}")
                    
            except Exception as e:
                logger.warning(f"Could not process imports in {py_file}: {e}")
        
        return {
            'processed_files': processed_files,
            'fixed_files': fixed_files
        }
    
    def generate_maintenance_report(self, 
                                  cleanup_results: Dict,
                                  git_results: Dict,
                                  health_results: Dict,
                                  doc_results: Dict,
                                  import_results: Dict) -> str:
        """Generate maintenance report."""
        
        report = []
        report.append("# Repository Maintenance Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Cleanup results
        report.append("## Cleanup Results")
        report.append(f"- Files removed: {cleanup_results.get('removed_files', 0)}")
        report.append(f"- Directories removed: {cleanup_results.get('removed_dirs', 0)}")
        report.append(f"- Space freed: {cleanup_results.get('freed_space_mb', 0):.2f} MB")
        report.append("")
        
        # Git optimization
        report.append("## Git Optimization")
        for operation, result in git_results.items():
            status = "✅" if result == "success" else "❌"
            report.append(f"- {operation}: {status} {result}")
        report.append("")
        
        # Repository health
        report.append("## Repository Health")
        if 'repository_size' in health_results:
            report.append(f"- Repository size: {health_results['repository_size']}")
        if 'large_files_count' in health_results:
            report.append(f"- Large files (>10MB): {health_results['large_files_count']}")
        if 'remote_branches' in health_results:
            report.append(f"- Remote branches: {health_results['remote_branches']}")
        
        if 'code_metrics' in health_results:
            code_metrics = health_results['code_metrics']
            if 'lines_of_code' in code_metrics:
                report.append(f"- Lines of code: {code_metrics['lines_of_code']}")
            if 'file_counts' in code_metrics:
                file_counts = code_metrics['file_counts']
                report.append(f"- Python files: {file_counts.get('.py', 0)}")
                report.append(f"- Documentation files: {file_counts.get('.md', 0)}")
        report.append("")
        
        # Documentation check
        report.append("## Documentation Check")
        report.append(f"- Links checked: {doc_results.get('checked_links', 0)}")
        if doc_results.get('broken_links', 0) > 0:
            report.append(f"- ⚠️ Broken links found: {doc_results['broken_links']}")
        else:
            report.append("- ✅ No broken links found")
        report.append("")
        
        # Import organization
        report.append("## Import Organization")
        report.append(f"- Files processed: {import_results.get('processed_files', 0)}")
        report.append(f"- Files fixed: {import_results.get('fixed_files', 0)}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        recommendations = []
        
        if cleanup_results.get('freed_space_mb', 0) > 100:
            recommendations.append("- Consider running cleanup more frequently to save disk space")
        
        if health_results.get('large_files_count', 0) > 5:
            recommendations.append("- Review large files and consider using Git LFS")
        
        if doc_results.get('broken_links', 0) > 0:
            recommendations.append("- Fix broken documentation links")
        
        if import_results.get('fixed_files', 0) > 10:
            recommendations.append("- Consider setting up pre-commit hooks for import sorting")
        
        if not recommendations:
            recommendations.append("- Repository is in good health!")
        
        for rec in recommendations:
            report.append(rec)
        
        return "\n".join(report)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Repository maintenance automation')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up temporary files')
    parser.add_argument('--optimize-git', action='store_true',
                        help='Optimize git repository')
    parser.add_argument('--check-health', action='store_true',
                        help='Check repository health')
    parser.add_argument('--check-docs', action='store_true',
                        help='Check documentation links')
    parser.add_argument('--organize-imports', action='store_true',
                        help='Organize Python imports')
    parser.add_argument('--all', action='store_true',
                        help='Run all maintenance tasks')
    parser.add_argument('--output-report', help='Output report file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.all:
        args.cleanup = True
        args.optimize_git = True
        args.check_health = True
        args.check_docs = True
        args.organize_imports = True
    
    maintainer = RepositoryMaintainer()
    
    # Initialize results
    cleanup_results = {}
    git_results = {}
    health_results = {}
    doc_results = {}
    import_results = {}
    
    try:
        # Run maintenance tasks
        if args.cleanup:
            cleanup_results = maintainer.cleanup_temporary_files()
            logger.info(f"Cleanup completed: {cleanup_results['freed_space_mb']:.2f} MB freed")
        
        if args.optimize_git:
            git_results = maintainer.optimize_git_repository()
            logger.info("Git optimization completed")
        
        if args.check_health:
            health_results = maintainer.check_repository_health()
            logger.info("Health check completed")
        
        if args.check_docs:
            doc_results = maintainer.update_documentation_links()
            logger.info("Documentation check completed")
        
        if args.organize_imports:
            import_results = maintainer.organize_imports()
            logger.info("Import organization completed")
        
        # Generate report
        if any([cleanup_results, git_results, health_results, doc_results, import_results]):
            report = maintainer.generate_maintenance_report(
                cleanup_results, git_results, health_results, doc_results, import_results
            )
            
            if args.output_report:
                with open(args.output_report, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {args.output_report}")
            
            print(report)
        else:
            print("No maintenance tasks specified. Use --help for options.")
        
        logger.info("Repository maintenance completed successfully")
        
    except Exception as e:
        logger.error(f"Error during repository maintenance: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()