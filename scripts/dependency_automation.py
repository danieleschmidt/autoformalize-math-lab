#!/usr/bin/env python3
"""
Dependency automation script for autoformalize-math-lab.

This script automates dependency management including updates, security checks,
and compatibility verification.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import requests
import semver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages project dependencies."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.requirements_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-dev.txt",
            self.project_root / "pyproject.toml"
        ]
        
    def check_security_vulnerabilities(self) -> Dict[str, List[Dict]]:
        """Check for security vulnerabilities in dependencies."""
        logger.info("Checking for security vulnerabilities...")
        
        vulnerabilities = {
            'safety': [],
            'pip_audit': [],
            'bandit': []
        }
        
        # Run safety check
        try:
            result = subprocess.run([
                'python', '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities['safety'] = safety_data
                except json.JSONDecodeError:
                    logger.warning("Could not parse safety output")
        except Exception as e:
            logger.error(f"Error running safety check: {e}")
        
        # Run pip-audit
        try:
            result = subprocess.run([
                'python', '-m', 'pip_audit', '--format=json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities['pip_audit'] = audit_data.get('vulnerabilities', [])
                except json.JSONDecodeError:
                    logger.warning("Could not parse pip-audit output")
        except Exception as e:
            logger.error(f"Error running pip-audit: {e}")
        
        # Run bandit for code security issues
        try:
            result = subprocess.run([
                'python', '-m', 'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    vulnerabilities['bandit'] = bandit_data.get('results', [])
                except json.JSONDecodeError:
                    logger.warning("Could not parse bandit output")
        except Exception as e:
            logger.error(f"Error running bandit: {e}")
        
        return vulnerabilities
    
    def check_outdated_dependencies(self) -> List[Dict[str, str]]:
        """Check for outdated dependencies."""
        logger.info("Checking for outdated dependencies...")
        
        try:
            result = subprocess.run([
                'python', '-m', 'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"Error checking outdated dependencies: {e}")
        
        return []
    
    def update_dependencies(self, security_only: bool = False) -> Dict[str, str]:
        """Update dependencies."""
        logger.info(f"Updating dependencies (security_only={security_only})...")
        
        updates = {}
        
        # Get list of outdated packages
        outdated = self.check_outdated_dependencies()
        
        if security_only:
            # Filter to only security-related updates
            security_vulns = self.check_security_vulnerabilities()
            vulnerable_packages = set()
            
            for vuln_list in security_vulns.values():
                for vuln in vuln_list:
                    if isinstance(vuln, dict) and 'package_name' in vuln:
                        vulnerable_packages.add(vuln['package_name'])
                    elif isinstance(vuln, dict) and 'name' in vuln:
                        vulnerable_packages.add(vuln['name'])
            
            outdated = [pkg for pkg in outdated if pkg['name'] in vulnerable_packages]
        
        # Update packages
        for package in outdated:
            package_name = package['name']
            current_version = package['version']
            latest_version = package['latest_version']
            
            try:
                logger.info(f"Updating {package_name} from {current_version} to {latest_version}")
                
                result = subprocess.run([
                    'python', '-m', 'pip', 'install', '--upgrade', package_name
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    updates[package_name] = f"{current_version} -> {latest_version}"
                    logger.info(f"Successfully updated {package_name}")
                else:
                    logger.error(f"Failed to update {package_name}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error updating {package_name}: {e}")
        
        return updates
    
    def test_compatibility(self) -> bool:
        """Test compatibility after dependency updates."""
        logger.info("Testing compatibility after updates...")
        
        try:
            # Run basic import tests
            result = subprocess.run([
                'python', '-c', 'import autoformalize; print("Import successful")'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"Import test failed: {result.stderr}")
                return False
            
            # Run unit tests
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/unit/', '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error("Unit tests failed after dependency update")
                return False
            
            logger.info("Compatibility tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error running compatibility tests: {e}")
            return False
    
    def generate_requirements_freeze(self) -> str:
        """Generate frozen requirements with exact versions."""
        try:
            result = subprocess.run([
                'python', '-m', 'pip', 'freeze'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.error(f"Error generating requirements freeze: {e}")
        
        return ""
    
    def backup_requirements(self) -> str:
        """Backup current requirements."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"requirements_backup_{timestamp}.txt"
        
        frozen_reqs = self.generate_requirements_freeze()
        if frozen_reqs:
            backup_path = self.project_root / backup_filename
            with open(backup_path, 'w') as f:
                f.write(frozen_reqs)
            logger.info(f"Requirements backed up to {backup_path}")
            return str(backup_path)
        
        return ""
    
    def restore_requirements(self, backup_file: str) -> bool:
        """Restore requirements from backup."""
        try:
            logger.info(f"Restoring requirements from {backup_file}")
            
            result = subprocess.run([
                'python', '-m', 'pip', 'install', '-r', backup_file
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("Requirements restored successfully")
                return True
            else:
                logger.error(f"Failed to restore requirements: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring requirements: {e}")
            return False

class LicenseChecker:
    """Checks license compatibility of dependencies."""
    
    COMPATIBLE_LICENSES = {
        'MIT', 'MIT License', 'BSD', 'BSD License', 'Apache', 'Apache 2.0',
        'Apache Software License', 'Python Software Foundation License',
        'ISC License', 'Mozilla Public License 2.0'
    }
    
    INCOMPATIBLE_LICENSES = {
        'GPL', 'GPL v2', 'GPL v3', 'AGPL', 'AGPL v3', 'Copyleft'
    }
    
    def __init__(self):
        self.license_issues = []
    
    def check_licenses(self) -> Dict[str, List[Dict]]:
        """Check license compatibility of all dependencies."""
        logger.info("Checking license compatibility...")
        
        try:
            result = subprocess.run([
                'python', '-m', 'pip_licenses', '--format=json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                licenses = json.loads(result.stdout)
                return self._analyze_licenses(licenses)
                
        except Exception as e:
            logger.error(f"Error checking licenses: {e}")
        
        return {'compatible': [], 'incompatible': [], 'unknown': []}
    
    def _analyze_licenses(self, licenses: List[Dict]) -> Dict[str, List[Dict]]:
        """Analyze license compatibility."""
        result = {
            'compatible': [],
            'incompatible': [],
            'unknown': []
        }
        
        for license_info in licenses:
            license_name = license_info.get('License', '').strip()
            package_name = license_info.get('Name', '')
            
            if self._is_compatible_license(license_name):
                result['compatible'].append(license_info)
            elif self._is_incompatible_license(license_name):
                result['incompatible'].append(license_info)
                logger.warning(f"Incompatible license found: {package_name} ({license_name})")
            else:
                result['unknown'].append(license_info)
                logger.info(f"Unknown license: {package_name} ({license_name})")
        
        return result
    
    def _is_compatible_license(self, license_name: str) -> bool:
        """Check if license is compatible."""
        return any(compatible in license_name for compatible in self.COMPATIBLE_LICENSES)
    
    def _is_incompatible_license(self, license_name: str) -> bool:
        """Check if license is incompatible."""
        return any(incompatible in license_name for incompatible in self.INCOMPATIBLE_LICENSES)

class AutomationReporter:
    """Generate reports for dependency automation."""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_security_report(self, vulnerabilities: Dict) -> str:
        """Generate security vulnerabilities report."""
        report = []
        report.append("# Security Vulnerabilities Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        total_vulns = sum(len(vulns) for vulns in vulnerabilities.values())
        report.append(f"Total vulnerabilities found: {total_vulns}")
        report.append("")
        
        for tool, vulns in vulnerabilities.items():
            if vulns:
                report.append(f"## {tool.title()} Results")
                report.append(f"Found {len(vulns)} issues:")
                report.append("")
                
                for vuln in vulns[:5]:  # Show first 5
                    if isinstance(vuln, dict):
                        package = vuln.get('package_name', vuln.get('name', 'Unknown'))
                        severity = vuln.get('severity', vuln.get('issue_severity', 'Unknown'))
                        description = vuln.get('advisory', vuln.get('issue_text', 'No description'))
                        
                        report.append(f"- **{package}** ({severity}): {description[:100]}...")
                
                if len(vulns) > 5:
                    report.append(f"... and {len(vulns) - 5} more issues")
                report.append("")
        
        return "\n".join(report)
    
    def generate_update_report(self, updates: Dict[str, str]) -> str:
        """Generate dependency updates report."""
        report = []
        report.append("# Dependency Updates Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        if updates:
            report.append(f"Updated {len(updates)} packages:")
            report.append("")
            
            for package, change in updates.items():
                report.append(f"- {package}: {change}")
        else:
            report.append("No packages were updated.")
        
        report.append("")
        return "\n".join(report)
    
    def generate_license_report(self, license_data: Dict) -> str:
        """Generate license compatibility report."""
        report = []
        report.append("# License Compatibility Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        compatible_count = len(license_data.get('compatible', []))
        incompatible_count = len(license_data.get('incompatible', []))
        unknown_count = len(license_data.get('unknown', []))
        
        report.append(f"- Compatible licenses: {compatible_count}")
        report.append(f"- Incompatible licenses: {incompatible_count}")
        report.append(f"- Unknown licenses: {unknown_count}")
        report.append("")
        
        if incompatible_count > 0:
            report.append("## ⚠️ Incompatible Licenses")
            for license_info in license_data['incompatible']:
                package = license_info.get('Name', 'Unknown')
                license_name = license_info.get('License', 'Unknown')
                report.append(f"- {package}: {license_name}")
            report.append("")
        
        if unknown_count > 0:
            report.append("## ❓ Unknown Licenses (Review Required)")
            for license_info in license_data['unknown'][:10]:  # Show first 10
                package = license_info.get('Name', 'Unknown')
                license_name = license_info.get('License', 'Unknown')
                report.append(f"- {package}: {license_name}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Automate dependency management')
    parser.add_argument('--check-security', action='store_true',
                        help='Check for security vulnerabilities')
    parser.add_argument('--check-outdated', action='store_true',
                        help='Check for outdated dependencies')
    parser.add_argument('--update', action='store_true',
                        help='Update dependencies')
    parser.add_argument('--security-only', action='store_true',
                        help='Only update packages with security vulnerabilities')
    parser.add_argument('--check-licenses', action='store_true',
                        help='Check license compatibility')
    parser.add_argument('--test-compatibility', action='store_true',
                        help='Test compatibility after updates')
    parser.add_argument('--output-dir', default='reports',
                        help='Output directory for reports')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    dep_manager = DependencyManager()
    license_checker = LicenseChecker()
    reporter = AutomationReporter()
    
    try:
        # Security check
        if args.check_security:
            logger.info("Running security vulnerability check...")
            vulnerabilities = dep_manager.check_security_vulnerabilities()
            
            security_report = reporter.generate_security_report(vulnerabilities)
            with open(output_dir / 'security_report.md', 'w') as f:
                f.write(security_report)
            
            print(security_report)
        
        # Outdated dependencies check
        if args.check_outdated:
            logger.info("Checking for outdated dependencies...")
            outdated = dep_manager.check_outdated_dependencies()
            
            if outdated:
                print(f"\nFound {len(outdated)} outdated dependencies:")
                for pkg in outdated:
                    print(f"- {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
            else:
                print("All dependencies are up to date!")
        
        # Update dependencies
        if args.update:
            # Backup current requirements
            backup_file = dep_manager.backup_requirements()
            
            try:
                updates = dep_manager.update_dependencies(args.security_only)
                
                if updates:
                    # Test compatibility
                    if args.test_compatibility:
                        if not dep_manager.test_compatibility():
                            logger.error("Compatibility tests failed, rolling back...")
                            if backup_file:
                                dep_manager.restore_requirements(backup_file)
                            sys.exit(1)
                    
                    # Generate update report
                    update_report = reporter.generate_update_report(updates)
                    with open(output_dir / 'update_report.md', 'w') as f:
                        f.write(update_report)
                    
                    print(update_report)
                else:
                    print("No updates performed.")
                    
            except Exception as e:
                logger.error(f"Error during update: {e}")
                if backup_file:
                    logger.info("Rolling back to previous state...")
                    dep_manager.restore_requirements(backup_file)
                sys.exit(1)
        
        # License check
        if args.check_licenses:
            logger.info("Checking license compatibility...")
            license_data = license_checker.check_licenses()
            
            license_report = reporter.generate_license_report(license_data)
            with open(output_dir / 'license_report.md', 'w') as f:
                f.write(license_report)
            
            print(license_report)
            
            # Exit with error if incompatible licenses found
            if license_data.get('incompatible'):
                logger.error("Incompatible licenses detected!")
                sys.exit(1)
        
        logger.info("Dependency automation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in dependency automation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()