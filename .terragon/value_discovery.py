#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine
Continuously discovers, scores, and executes highest-value SDLC improvements
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

class ValueDiscoveryEngine:
    """Core engine for discovering and scoring value opportunities"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config = self._load_config(config_path)
        self.repo_root = Path.cwd()
        self.metrics_file = self.repo_root / ".terragon" / "value-metrics.json"
        self.backlog_file = self.repo_root / "BACKLOG.md"
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            "scoring": {
                "weights": {"maturing": {"wsjf": 0.5, "ice": 0.2, "technicalDebt": 0.2, "security": 0.1}},
                "thresholds": {"minScore": 15, "securityBoost": 2.0}
            },
            "discovery": {"sources": ["gitHistory", "staticAnalysis"]},
            "execution": {"maxConcurrentTasks": 1, "branchPrefix": "auto-value"}
        }

    def discover_value_opportunities(self) -> List[Dict]:
        """Main discovery method - finds all value opportunities"""
        opportunities = []
        
        # Discover from multiple sources
        opportunities.extend(self._discover_from_git_history())
        opportunities.extend(self._discover_from_static_analysis())
        opportunities.extend(self._discover_from_security_scans())
        opportunities.extend(self._discover_from_dependencies())
        opportunities.extend(self._discover_from_performance())
        opportunities.extend(self._discover_from_documentation())
        
        # Score and rank opportunities
        scored_opportunities = [self._score_opportunity(opp) for opp in opportunities]
        
        # Filter and sort by composite score
        filtered = [opp for opp in scored_opportunities if opp.get('compositeScore', 0) >= self.config['scoring']['thresholds']['minScore']]
        return sorted(filtered, key=lambda x: x.get('compositeScore', 0), reverse=True)

    def _discover_from_git_history(self) -> List[Dict]:
        """Discover technical debt from git history and comments"""
        opportunities = []
        
        try:
            # Find TODO, FIXME, HACK comments
            result = subprocess.run(['grep', '-r', '-n', '-i', 
                                   '--include=*.py', '--include=*.md', '--include=*.yml',
                                   r'TODO\|FIXME\|HACK\|XXX\|BUG'], 
                                  cwd=self.repo_root, capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        opportunities.append({
                            'id': f"comment-{hash(line)}",
                            'title': f"Address code comment: {parts[2][:50]}...",
                            'type': 'technical-debt',
                            'description': parts[2].strip(),
                            'file': parts[0],
                            'line': parts[1],
                            'effort': self._estimate_effort_from_comment(parts[2]),
                            'priority': 'medium',
                            'source': 'git-history'
                        })
        except Exception as e:
            print(f"Warning: Git history analysis failed: {e}")
            
        return opportunities

    def _discover_from_static_analysis(self) -> List[Dict]:
        """Discover issues from static analysis tools"""
        opportunities = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run(['ruff', 'check', 'src/', '--output-format=json'], 
                                  capture_output=True, text=True, cwd=self.repo_root)
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues[:10]:  # Limit to top 10 issues
                    opportunities.append({
                        'id': f"ruff-{issue.get('code', 'unknown')}",
                        'title': f"Fix code quality: {issue.get('message', 'Unknown issue')}",
                        'type': 'code-quality',
                        'description': issue.get('message', ''),
                        'file': issue.get('filename', ''),
                        'line': issue.get('location', {}).get('row', 0),
                        'effort': 1,
                        'priority': 'low',
                        'source': 'static-analysis'
                    })
        except Exception as e:
            print(f"Warning: Ruff analysis failed: {e}")
            
        return opportunities

    def _discover_from_security_scans(self) -> List[Dict]:
        """Discover security vulnerabilities"""
        opportunities = []
        
        # Run safety check for dependency vulnerabilities
        try:
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True, cwd=self.repo_root)
            if result.stdout:
                safety_report = json.loads(result.stdout)
                for vuln in safety_report.get('vulnerabilities', [])[:5]:
                    opportunities.append({
                        'id': f"security-{vuln.get('id', 'unknown')}",
                        'title': f"Fix security vulnerability in {vuln.get('package_name', 'unknown')}",
                        'type': 'security',
                        'description': vuln.get('advisory', ''),
                        'effort': 2,
                        'priority': 'high',
                        'securitySeverity': vuln.get('severity', 'medium'),
                        'source': 'security-scan'
                    })
        except Exception as e:
            print(f"Warning: Security scan failed: {e}")
            
        return opportunities

    def _discover_from_dependencies(self) -> List[Dict]:
        """Discover outdated dependencies"""
        opportunities = []
        
        try:
            # Check for outdated pip packages
            result = subprocess.run(['pip', 'list', '--outdated', '--format=json'], 
                                  capture_output=True, text=True)
            if result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:5]:  # Limit to top 5 outdated packages
                    opportunities.append({
                        'id': f"dependency-{pkg['name']}",
                        'title': f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        'type': 'dependency-update',
                        'description': f"Package {pkg['name']} is outdated",
                        'effort': 1,
                        'priority': 'low',
                        'package': pkg['name'],
                        'currentVersion': pkg['version'],
                        'latestVersion': pkg['latest_version'],
                        'source': 'dependency-check'
                    })
        except Exception as e:
            print(f"Warning: Dependency check failed: {e}")
            
        return opportunities

    def _discover_from_performance(self) -> List[Dict]:
        """Discover performance optimization opportunities"""
        opportunities = []
        
        # Look for common performance anti-patterns in Python code
        performance_patterns = [
            (r'\.append\(.*\)\s*in\s+.*for', 'List comprehension optimization opportunity'),
            (r'len\([^)]+\)\s*==\s*0', 'Use "not list" instead of "len(list) == 0"'),
            (r'\.keys\(\)\s+in\s+', 'Direct dict membership check optimization')
        ]
        
        try:
            for root, dirs, files in os.walk(self.repo_root / "src"):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                for pattern, description in performance_patterns:
                                    matches = re.finditer(pattern, content, re.IGNORECASE)
                                    for match in matches:
                                        line_num = content[:match.start()].count('\n') + 1
                                        opportunities.append({
                                            'id': f"perf-{hash(f'{file_path}:{line_num}')}",
                                            'title': f"Performance optimization in {file}",
                                            'type': 'performance',
                                            'description': description,
                                            'file': str(file_path.relative_to(self.repo_root)),
                                            'line': line_num,
                                            'effort': 1,
                                            'priority': 'medium',
                                            'source': 'performance-analysis'
                                        })
                        except Exception:
                            continue
        except Exception as e:
            print(f"Warning: Performance analysis failed: {e}")
            
        return opportunities

    def _discover_from_documentation(self) -> List[Dict]:
        """Discover documentation gaps and improvements"""
        opportunities = []
        
        # Check for functions without docstrings
        try:
            result = subprocess.run(['python', '-c', '''
import ast
import os
import sys

def find_undocumented_functions(file_path):
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
        
        undocumented = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    undocumented.append((node.name, node.lineno))
        return undocumented
    except:
        return []

for root, dirs, files in os.walk("src/"):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            undocumented = find_undocumented_functions(file_path)
            for func_name, line_num in undocumented[:3]:  # Limit to 3 per file
                print(f"{file_path}:{line_num}:{func_name}")
'''], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        opportunities.append({
                            'id': f"docs-{hash(line)}",
                            'title': f"Add docstring to function {parts[2]}",
                            'type': 'documentation',
                            'description': f"Function {parts[2]} lacks documentation",
                            'file': parts[0],
                            'line': int(parts[1]),
                            'effort': 1,
                            'priority': 'low',
                            'source': 'documentation-analysis'
                        })
        except Exception as e:
            print(f"Warning: Documentation analysis failed: {e}")
            
        return opportunities

    def _estimate_effort_from_comment(self, comment: str) -> int:
        """Estimate effort based on comment content"""
        comment_lower = comment.lower()
        if any(word in comment_lower for word in ['complex', 'refactor', 'rewrite']):
            return 5
        elif any(word in comment_lower for word in ['fix', 'update', 'change']):
            return 2
        else:
            return 1

    def _score_opportunity(self, opportunity: Dict) -> Dict:
        """Calculate comprehensive scoring for an opportunity"""
        # WSJF Components
        user_business_value = self._calculate_business_value(opportunity)
        time_criticality = self._calculate_time_criticality(opportunity)
        risk_reduction = self._calculate_risk_reduction(opportunity)
        opportunity_enablement = self._calculate_opportunity_enablement(opportunity)
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = opportunity.get('effort', 1)
        wsjf = cost_of_delay / max(job_size, 0.1)
        
        # ICE Components
        impact = self._calculate_impact(opportunity)
        confidence = self._calculate_confidence(opportunity)
        ease = 10 - min(job_size, 10)  # Inverse of effort
        ice = impact * confidence * ease
        
        # Technical Debt Score
        tech_debt_score = self._calculate_tech_debt_score(opportunity)
        
        # Apply category-specific boosts
        category_multiplier = self._get_category_multiplier(opportunity.get('type', 'other'))
        
        # Composite Score
        weights = self.config['scoring']['weights']['maturing']
        composite_score = (
            weights['wsjf'] * wsjf +
            weights['ice'] * (ice / 100) +  # Normalize ICE to similar scale
            weights['technicalDebt'] * tech_debt_score +
            weights['security'] * (2.0 if opportunity.get('type') == 'security' else 1.0)
        ) * category_multiplier
        
        # Update opportunity with scores
        opportunity.update({
            'wsjf': round(wsjf, 2),
            'ice': round(ice, 2),
            'techDebtScore': round(tech_debt_score, 2),
            'compositeScore': round(composite_score, 2),
            'impact': impact,
            'confidence': confidence,
            'ease': ease,
            'costOfDelay': cost_of_delay,
            'jobSize': job_size
        })
        
        return opportunity

    def _calculate_business_value(self, opportunity: Dict) -> float:
        """Calculate business value component"""
        type_values = {
            'security': 9,
            'performance': 7,
            'technical-debt': 5,
            'code-quality': 4,
            'documentation': 3,
            'dependency-update': 2
        }
        return type_values.get(opportunity.get('type'), 3)

    def _calculate_time_criticality(self, opportunity: Dict) -> float:
        """Calculate time criticality"""
        if opportunity.get('type') == 'security':
            return 8
        elif opportunity.get('priority') == 'high':
            return 6
        elif opportunity.get('priority') == 'medium':
            return 4
        else:
            return 2

    def _calculate_risk_reduction(self, opportunity: Dict) -> float:
        """Calculate risk reduction value"""
        if opportunity.get('type') == 'security':
            return 9
        elif opportunity.get('type') == 'technical-debt':
            return 6
        elif opportunity.get('type') == 'performance':
            return 4
        else:
            return 2

    def _calculate_opportunity_enablement(self, opportunity: Dict) -> float:
        """Calculate opportunity enablement value"""
        if opportunity.get('type') in ['technical-debt', 'code-quality']:
            return 5
        elif opportunity.get('type') == 'documentation':
            return 3
        else:
            return 2

    def _calculate_impact(self, opportunity: Dict) -> int:
        """Calculate ICE impact (1-10)"""
        type_impacts = {
            'security': 9,
            'performance': 8,
            'technical-debt': 6,
            'code-quality': 5,
            'documentation': 4,
            'dependency-update': 3
        }
        return type_impacts.get(opportunity.get('type'), 4)

    def _calculate_confidence(self, opportunity: Dict) -> int:
        """Calculate ICE confidence (1-10)"""
        if opportunity.get('source') in ['static-analysis', 'security-scan']:
            return 9
        elif opportunity.get('source') == 'git-history':
            return 7
        else:
            return 6

    def _calculate_tech_debt_score(self, opportunity: Dict) -> float:
        """Calculate technical debt score"""
        if opportunity.get('type') == 'technical-debt':
            return 8
        elif opportunity.get('type') == 'code-quality':
            return 6
        elif opportunity.get('type') == 'performance':
            return 4
        else:
            return 2

    def _get_category_multiplier(self, opportunity_type: str) -> float:
        """Get category-specific multiplier"""
        categories = self.config.get('categories', {})
        for category, config in categories.items():
            if opportunity_type.startswith(category) or category in opportunity_type:
                return config.get('multiplier', 1.0)
        return 1.0

    def generate_backlog(self, opportunities: List[Dict]) -> None:
        """Generate and update BACKLOG.md with discovered opportunities"""
        now = datetime.now()
        
        content = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Next Execution: {(now + timedelta(hours=1)).isoformat()}

## 🎯 Next Best Value Item
"""
        
        if opportunities:
            best = opportunities[0]
            content += f"""**[{best['id']}] {best['title']}**
- **Composite Score**: {best.get('compositeScore', 0)}
- **WSJF**: {best.get('wsjf', 0)} | **ICE**: {best.get('ice', 0)} | **Tech Debt**: {best.get('techDebtScore', 0)}
- **Estimated Effort**: {best.get('effort', 1)} hours
- **Type**: {best.get('type', 'unknown')}
- **Priority**: {best.get('priority', 'medium')}

"""

        content += """## 📋 Top Value Opportunities

| Rank | ID | Title | Score | Type | Effort |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, opp in enumerate(opportunities[:10], 1):
            content += f"| {i} | {opp['id'][:12]} | {opp['title'][:50]} | {opp.get('compositeScore', 0):.1f} | {opp.get('type', 'unknown')} | {opp.get('effort', 1)}h |\n"
        
        content += f"""

## 📈 Value Metrics
- **Total Opportunities**: {len(opportunities)}
- **Average Score**: {sum(opp.get('compositeScore', 0) for opp in opportunities) / len(opportunities) if opportunities else 0:.1f}
- **High Priority Items**: {len([opp for opp in opportunities if opp.get('priority') == 'high'])}
- **Security Items**: {len([opp for opp in opportunities if opp.get('type') == 'security'])}

## 🔄 Discovery Sources Breakdown
- **Static Analysis**: {len([opp for opp in opportunities if opp.get('source') == 'static-analysis'])}
- **Security Scans**: {len([opp for opp in opportunities if opp.get('source') == 'security-scan'])}
- **Git History**: {len([opp for opp in opportunities if opp.get('source') == 'git-history'])}
- **Performance Analysis**: {len([opp for opp in opportunities if opp.get('source') == 'performance-analysis'])}
- **Documentation**: {len([opp for opp in opportunities if opp.get('source') == 'documentation-analysis'])}
- **Dependencies**: {len([opp for opp in opportunities if opp.get('source') == 'dependency-check'])}

Generated by Terragon Autonomous SDLC Enhancement System
"""
        
        with open(self.backlog_file, 'w') as f:
            f.write(content)

    def save_metrics(self, opportunities: List[Dict]) -> None:
        """Save value metrics to JSON file"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "totalOpportunities": len(opportunities),
            "averageScore": sum(opp.get('compositeScore', 0) for opp in opportunities) / len(opportunities) if opportunities else 0,
            "topOpportunities": opportunities[:5],
            "categoryBreakdown": {},
            "sourceBreakdown": {},
            "priorityBreakdown": {}
        }
        
        # Calculate breakdowns
        for opp in opportunities:
            # Category breakdown
            category = opp.get('type', 'unknown')
            metrics['categoryBreakdown'][category] = metrics['categoryBreakdown'].get(category, 0) + 1
            
            # Source breakdown
            source = opp.get('source', 'unknown')
            metrics['sourceBreakdown'][source] = metrics['sourceBreakdown'].get(source, 0) + 1
            
            # Priority breakdown
            priority = opp.get('priority', 'unknown')
            metrics['priorityBreakdown'][priority] = metrics['priorityBreakdown'].get(priority, 0) + 1
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    """Main execution function"""
    print("🚀 Starting Terragon Autonomous Value Discovery...")
    
    engine = ValueDiscoveryEngine()
    opportunities = engine.discover_value_opportunities()
    
    print(f"📊 Discovered {len(opportunities)} value opportunities")
    
    if opportunities:
        print(f"🎯 Top opportunity: {opportunities[0]['title']} (Score: {opportunities[0].get('compositeScore', 0):.1f})")
        
        # Generate backlog and save metrics
        engine.generate_backlog(opportunities)
        engine.save_metrics(opportunities)
        
        print(f"📝 Updated BACKLOG.md with {len(opportunities)} opportunities")
        print("✅ Value discovery complete!")
    else:
        print("ℹ️  No value opportunities discovered at this time")

if __name__ == "__main__":
    main()