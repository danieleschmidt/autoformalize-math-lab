"""Security audit and validation tools.

This module provides security scanning and validation capabilities
for the autoformalize system.
"""

import re
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.logging_config import setup_logger


class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityFinding:
    """Represents a security finding."""
    level: SecurityLevel
    category: str
    description: str
    location: str
    recommendation: str
    cwe_id: Optional[str] = None


class SecurityAuditor:
    """Comprehensive security auditor for the autoformalize system."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.findings: List[SecurityFinding] = []
    
    def audit_input_validation(self, content: str) -> List[SecurityFinding]:
        """Audit input validation for potential injection attacks."""
        findings = []
        
        # Check for LaTeX injection patterns
        dangerous_commands = [
            r'\\input\{.*\}',
            r'\\include\{.*\}',
            r'\\write\d+',
            r'\\immediate',
            r'\\catcode',
            r'\\openin',
            r'\\openout',
            r'\\read',
            r'\\write',
            r'\\expandafter',
            r'\\csname.*\\endcsname'
        ]
        
        for pattern in dangerous_commands:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(SecurityFinding(
                    level=SecurityLevel.HIGH,
                    category="Injection",
                    description=f"Potentially dangerous LaTeX command: {match.group()}",
                    location=f"Position {match.start()}-{match.end()}",
                    recommendation="Sanitize or block dangerous LaTeX commands",
                    cwe_id="CWE-94"
                ))
        
        # Check for oversized input
        if len(content) > 1000000:  # 1MB limit
            findings.append(SecurityFinding(
                level=SecurityLevel.MEDIUM,
                category="DoS",
                description=f"Input size {len(content)} bytes exceeds recommended limit",
                location="Input size",
                recommendation="Implement input size limits",
                cwe_id="CWE-400"
            ))
        
        # Check for suspicious Unicode characters
        suspicious_chars = [
            '\u202e',  # Right-to-left override
            '\u200e',  # Left-to-right mark
            '\u200f',  # Right-to-left mark
            '\ufeff',  # Byte order mark
        ]
        
        for char in suspicious_chars:
            if char in content:
                findings.append(SecurityFinding(
                    level=SecurityLevel.MEDIUM,
                    category="Obfuscation",
                    description=f"Suspicious Unicode character found: U+{ord(char):04X}",
                    location="Input content",
                    recommendation="Filter or escape suspicious Unicode characters",
                    cwe_id="CWE-116"
                ))
        
        return findings
    
    def audit_code_generation(self, generated_code: str, target_system: str) -> List[SecurityFinding]:
        """Audit generated formal code for security issues."""
        findings = []
        
        # System-specific security checks
        if target_system == "lean4":
            findings.extend(self._audit_lean4_code(generated_code))
        elif target_system == "isabelle":
            findings.extend(self._audit_isabelle_code(generated_code))
        elif target_system == "coq":
            findings.extend(self._audit_coq_code(generated_code))
        
        # Generic checks
        findings.extend(self._audit_generic_code(generated_code))
        
        return findings
    
    def _audit_lean4_code(self, code: str) -> List[SecurityFinding]:
        """Audit Lean 4 specific security issues."""
        findings = []
        
        # Check for unsafe imports
        unsafe_imports = [
            'System.IO',
            'System.Process',
            'System.File',
        ]
        
        for unsafe_import in unsafe_imports:
            if f"import {unsafe_import}" in code:
                findings.append(SecurityFinding(
                    level=SecurityLevel.HIGH,
                    category="Unsafe Import",
                    description=f"Potentially unsafe import: {unsafe_import}",
                    location="Import section",
                    recommendation="Remove unsafe system imports",
                    cwe_id="CWE-470"
                ))
        
        # Check for unsafe tactics
        unsafe_tactics = [
            'sorry',
            'admit',
            'unsafe',
        ]
        
        for tactic in unsafe_tactics:
            if tactic in code:
                findings.append(SecurityFinding(
                    level=SecurityLevel.MEDIUM,
                    category="Unsafe Proof",
                    description=f"Unsafe proof tactic used: {tactic}",
                    location="Proof section",
                    recommendation="Complete proofs without unsafe tactics",
                    cwe_id="CWE-670"
                ))
        
        return findings
    
    def _audit_isabelle_code(self, code: str) -> List[SecurityFinding]:
        """Audit Isabelle/HOL specific security issues."""
        findings = []
        
        # Check for ML code execution
        if 'ML {*' in code or 'ML_file' in code:
            findings.append(SecurityFinding(
                level=SecurityLevel.HIGH,
                category="Code Execution",
                description="ML code execution detected",
                location="ML section",
                recommendation="Avoid ML code in generated proofs",
                cwe_id="CWE-94"
            ))
        
        return findings
    
    def _audit_coq_code(self, code: str) -> List[SecurityFinding]:
        """Audit Coq specific security issues."""
        findings = []
        
        # Check for axioms (which could be unsound)
        if 'Axiom' in code:
            findings.append(SecurityFinding(
                level=SecurityLevel.MEDIUM,
                category="Soundness",
                description="Axiom declaration found",
                location="Axiom section",
                recommendation="Avoid axioms in generated code unless necessary",
                cwe_id="CWE-670"
            ))
        
        return findings
    
    def _audit_generic_code(self, code: str) -> List[SecurityFinding]:
        """Audit generic security issues in generated code."""
        findings = []
        
        # Check for excessive complexity
        lines = code.split('\n')
        if len(lines) > 1000:
            findings.append(SecurityFinding(
                level=SecurityLevel.LOW,
                category="Complexity",
                description=f"Generated code has {len(lines)} lines, may be too complex",
                location="Overall structure",
                recommendation="Consider breaking down complex proofs",
                cwe_id="CWE-1120"
            ))
        
        # Check for potential resource exhaustion patterns
        resource_patterns = [
            r'(\w+\s*::\s*)*\w+\s*->\s*(\w+\s*::\s*)*\w+\s*->\s*.*->\s*.*->\s*.*',  # Deep recursion pattern
            r'List\.repeat\s+\w+\s+\d{4,}',  # Large list creation
            r'Array\.make\s+\d{4,}',  # Large array creation
        ]
        
        for pattern in resource_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                findings.append(SecurityFinding(
                    level=SecurityLevel.MEDIUM,
                    category="Resource Exhaustion",
                    description=f"Potential resource exhaustion pattern: {match.group()}",
                    location=f"Line containing: {match.group()[:50]}...",
                    recommendation="Review resource usage patterns",
                    cwe_id="CWE-400"
                ))
        
        return findings
    
    def audit_api_security(self, api_request: Dict[str, Any]) -> List[SecurityFinding]:
        """Audit API request for security issues."""
        findings = []
        
        # Check request size
        request_size = len(str(api_request))
        if request_size > 10000000:  # 10MB
            findings.append(SecurityFinding(
                level=SecurityLevel.HIGH,
                category="DoS",
                description=f"API request size {request_size} bytes is excessive",
                location="Request body",
                recommendation="Implement request size limits",
                cwe_id="CWE-400"
            ))
        
        # Check for SQL injection patterns (even though we don't use SQL directly)
        if 'latex_content' in api_request:
            content = api_request['latex_content']
            sql_patterns = [
                r"'.*OR.*'",
                r"';.*--",
                r"UNION.*SELECT",
                r"DROP.*TABLE",
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    findings.append(SecurityFinding(
                        level=SecurityLevel.MEDIUM,
                        category="Injection",
                        description=f"Potential injection pattern detected: {pattern}",
                        location="LaTeX content",
                        recommendation="Implement input sanitization",
                        cwe_id="CWE-89"
                    ))
        
        # Check model parameter validation
        if 'model' in api_request:
            model = api_request['model']
            allowed_models = ['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus']
            if model not in allowed_models:
                findings.append(SecurityFinding(
                    level=SecurityLevel.MEDIUM,
                    category="Validation",
                    description=f"Unknown model specified: {model}",
                    location="Model parameter",
                    recommendation="Validate model parameter against allowed list",
                    cwe_id="CWE-20"
                ))
        
        return findings
    
    def audit_dependencies(self, requirements_file: Path) -> List[SecurityFinding]:
        """Audit Python dependencies for known vulnerabilities."""
        findings = []
        
        if not requirements_file.exists():
            findings.append(SecurityFinding(
                level=SecurityLevel.LOW,
                category="Configuration",
                description="Requirements file not found",
                location=str(requirements_file),
                recommendation="Maintain a requirements.txt file",
                cwe_id="CWE-1104"
            ))
            return findings
        
        try:
            # This would integrate with safety or similar tools in production
            content = requirements_file.read_text()
            
            # Check for pinned versions
            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
            unpinned_deps = []
            
            for line in lines:
                if '==' not in line and '>=' not in line and '~=' not in line:
                    unpinned_deps.append(line)
            
            if unpinned_deps:
                findings.append(SecurityFinding(
                    level=SecurityLevel.MEDIUM,
                    category="Dependency Management",
                    description=f"Unpinned dependencies: {', '.join(unpinned_deps)}",
                    location="requirements.txt",
                    recommendation="Pin all dependency versions",
                    cwe_id="CWE-1104"
                ))
            
            # Check for known vulnerable packages (example list)
            vulnerable_patterns = [
                r'requests<2\.25\.0',
                r'urllib3<1\.26\.0',
                r'pillow<8\.1\.1',
            ]
            
            for pattern in vulnerable_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    findings.append(SecurityFinding(
                        level=SecurityLevel.HIGH,
                        category="Vulnerable Dependency",
                        description=f"Potentially vulnerable dependency pattern: {pattern}",
                        location="requirements.txt",
                        recommendation="Update to secure versions",
                        cwe_id="CWE-1035"
                    ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                level=SecurityLevel.LOW,
                category="Audit Error",
                description=f"Failed to audit dependencies: {e}",
                location=str(requirements_file),
                recommendation="Investigate dependency audit failure",
                cwe_id="CWE-1104"
            ))
        
        return findings
    
    def audit_configuration(self, config: Dict[str, Any]) -> List[SecurityFinding]:
        """Audit configuration for security issues."""
        findings = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'api_key.*=.*["\'][^"\']{20,}["\']',
            r'password.*=.*["\'][^"\']+["\']',
            r'secret.*=.*["\'][^"\']+["\']',
            r'token.*=.*["\'][^"\']{20,}["\']',
        ]
        
        config_str = str(config)
        for pattern in secret_patterns:
            matches = re.finditer(pattern, config_str, re.IGNORECASE)
            for match in matches:
                findings.append(SecurityFinding(
                    level=SecurityLevel.CRITICAL,
                    category="Secrets Management",
                    description="Hardcoded secret detected in configuration",
                    location="Configuration",
                    recommendation="Use environment variables or secure secret management",
                    cwe_id="CWE-798"
                ))
        
        # Check for insecure defaults
        if config.get('debug', False):
            findings.append(SecurityFinding(
                level=SecurityLevel.MEDIUM,
                category="Configuration",
                description="Debug mode enabled",
                location="Debug setting",
                recommendation="Disable debug mode in production",
                cwe_id="CWE-489"
            ))
        
        # Check for excessive timeouts
        timeout = config.get('timeout', 0)
        if timeout > 300:  # 5 minutes
            findings.append(SecurityFinding(
                level=SecurityLevel.LOW,
                category="DoS",
                description=f"Timeout set to {timeout} seconds, may allow resource exhaustion",
                location="Timeout configuration",
                recommendation="Set reasonable timeout limits",
                cwe_id="CWE-400"
            ))
        
        return findings
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        findings_by_level = {level: [] for level in SecurityLevel}
        findings_by_category = {}
        
        for finding in self.findings:
            findings_by_level[finding.level].append(finding)
            
            if finding.category not in findings_by_category:
                findings_by_category[finding.category] = []
            findings_by_category[finding.category].append(finding)
        
        # Calculate risk score
        risk_score = (
            len(findings_by_level[SecurityLevel.CRITICAL]) * 10 +
            len(findings_by_level[SecurityLevel.HIGH]) * 7 +
            len(findings_by_level[SecurityLevel.MEDIUM]) * 4 +
            len(findings_by_level[SecurityLevel.LOW]) * 1
        )
        
        return {
            "summary": {
                "total_findings": len(self.findings),
                "risk_score": risk_score,
                "critical_count": len(findings_by_level[SecurityLevel.CRITICAL]),
                "high_count": len(findings_by_level[SecurityLevel.HIGH]),
                "medium_count": len(findings_by_level[SecurityLevel.MEDIUM]),
                "low_count": len(findings_by_level[SecurityLevel.LOW])
            },
            "findings_by_level": {
                level.value: [
                    {
                        "category": f.category,
                        "description": f.description,
                        "location": f.location,
                        "recommendation": f.recommendation,
                        "cwe_id": f.cwe_id
                    }
                    for f in findings
                ]
                for level, findings in findings_by_level.items()
            },
            "findings_by_category": {
                category: len(findings)
                for category, findings in findings_by_category.items()
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized security recommendations."""
        recommendations = []
        
        critical_count = len([f for f in self.findings if f.level == SecurityLevel.CRITICAL])
        high_count = len([f for f in self.findings if f.level == SecurityLevel.HIGH])
        
        if critical_count > 0:
            recommendations.append("🚨 URGENT: Address all critical security findings immediately")
        
        if high_count > 0:
            recommendations.append("⚠️ HIGH PRIORITY: Resolve high-severity security issues")
        
        # Category-specific recommendations
        categories = {f.category for f in self.findings}
        
        if "Injection" in categories:
            recommendations.append("Implement comprehensive input validation and sanitization")
        
        if "Secrets Management" in categories:
            recommendations.append("Migrate all secrets to secure environment variables or key management")
        
        if "DoS" in categories:
            recommendations.append("Implement rate limiting and resource usage controls")
        
        if "Vulnerable Dependency" in categories:
            recommendations.append("Update all dependencies to secure versions")
        
        recommendations.extend([
            "Establish regular security scanning in CI/CD pipeline",
            "Implement security monitoring and alerting",
            "Conduct regular penetration testing",
            "Maintain security documentation and incident response plan"
        ])
        
        return recommendations
    
    def run_comprehensive_audit(
        self,
        input_content: str,
        generated_code: str,
        target_system: str,
        api_request: Dict[str, Any],
        config: Dict[str, Any],
        requirements_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        self.findings.clear()
        
        # Run all audit components
        self.findings.extend(self.audit_input_validation(input_content))
        self.findings.extend(self.audit_code_generation(generated_code, target_system))
        self.findings.extend(self.audit_api_security(api_request))
        self.findings.extend(self.audit_configuration(config))
        
        if requirements_file:
            self.findings.extend(self.audit_dependencies(requirements_file))
        
        return self.generate_security_report()


def run_security_scan():
    """Run security scan from command line."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python security_audit.py <project_root>")
        return
    
    project_root = Path(sys.argv[1])
    auditor = SecurityAuditor()
    
    # Sample audit (in real implementation, would scan actual files)
    sample_config = {
        "debug": False,
        "timeout": 60,
        "model": "gpt-4"
    }
    
    sample_request = {
        "latex_content": "\\theorem{test}",
        "target_system": "lean4",
        "model": "gpt-4"
    }
    
    requirements_file = project_root / "requirements.txt"
    
    report = auditor.run_comprehensive_audit(
        input_content="\\theorem{test}",
        generated_code="theorem test : True := by trivial",
        target_system="lean4",
        api_request=sample_request,
        config=sample_config,
        requirements_file=requirements_file if requirements_file.exists() else None
    )
    
    print("🔒 Security Audit Report")
    print("=" * 50)
    print(f"Total Findings: {report['summary']['total_findings']}")
    print(f"Risk Score: {report['summary']['risk_score']}")
    print(f"Critical: {report['summary']['critical_count']}")
    print(f"High: {report['summary']['high_count']}")
    print(f"Medium: {report['summary']['medium_count']}")
    print(f"Low: {report['summary']['low_count']}")
    print()
    
    if report['recommendations']:
        print("📋 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")


if __name__ == "__main__":
    run_security_scan()