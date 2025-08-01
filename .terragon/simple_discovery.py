#!/usr/bin/env python3
"""
Simplified Autonomous Value Discovery Engine
Discovers and scores SDLC improvement opportunities using only standard library
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

class SimpleValueDiscovery:
    """Simplified value discovery engine"""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.metrics_file = self.repo_root / ".terragon" / "value-metrics.json"
        self.backlog_file = self.repo_root / "BACKLOG.md"
        
    def discover_opportunities(self) -> List[Dict]:
        """Discover value opportunities from multiple sources"""
        opportunities = []
        
        # Discover from code comments
        opportunities.extend(self._find_todo_comments())
        
        # Discover missing docstrings
        opportunities.extend(self._find_missing_docstrings())
        
        # Discover potential performance issues
        opportunities.extend(self._find_performance_issues())
        
        # Discover missing tests
        opportunities.extend(self._find_missing_tests())
        
        # Score and sort opportunities
        scored = [self._score_opportunity(opp) for opp in opportunities]
        return sorted(scored, key=lambda x: x.get('score', 0), reverse=True)
    
    def _find_todo_comments(self) -> List[Dict]:
        """Find TODO, FIXME, HACK comments in code"""
        opportunities = []
        comment_patterns = [
            (r'TODO:?\s*(.+)', 'TODO'),
            (r'FIXME:?\s*(.+)', 'FIXME'), 
            (r'HACK:?\s*(.+)', 'HACK'),
            (r'XXX:?\s*(.+)', 'XXX')
        ]
        
        for root, dirs, files in os.walk(self.repo_root):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.md', '.yml', '.yaml', '.js', '.ts')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                for pattern, comment_type in comment_patterns:
                                    match = re.search(pattern, line, re.IGNORECASE)
                                    if match:
                                        opportunities.append({
                                            'id': f"comment-{hash(f'{file_path}:{line_num}')}",
                                            'title': f"Address {comment_type}: {match.group(1)[:50]}...",
                                            'type': 'technical-debt',
                                            'description': match.group(1).strip(),
                                            'file': str(file_path.relative_to(self.repo_root)),
                                            'line': line_num,
                                            'effort': self._estimate_comment_effort(match.group(1)),
                                            'priority': self._get_comment_priority(comment_type),
                                            'source': 'code-comments'
                                        })
                    except (UnicodeDecodeError, IOError):
                        continue
        
        return opportunities
    
    def _find_missing_docstrings(self) -> List[Dict]:
        """Find functions without docstrings"""
        opportunities = []
        
        for root, dirs, files in os.walk(self.repo_root / "src"):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Simple regex to find function definitions
                        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
                        for match in re.finditer(func_pattern, content):
                            func_name = match.group(1)
                            # Skip private functions and special methods
                            if func_name.startswith('_'):
                                continue
                                
                            # Check if next non-empty line starts with triple quotes
                            lines = content[match.end():].split('\n')
                            has_docstring = False
                            for line in lines[:3]:
                                stripped = line.strip()
                                if stripped and (stripped.startswith('"""') or stripped.startswith("'''")):
                                    has_docstring = True
                                    break
                                elif stripped and not stripped.startswith('#'):
                                    break
                            
                            if not has_docstring:
                                line_num = content[:match.start()].count('\n') + 1
                                opportunities.append({
                                    'id': f"docstring-{hash(f'{file_path}:{func_name}')}",
                                    'title': f"Add docstring to function {func_name}",
                                    'type': 'documentation',
                                    'description': f"Function {func_name} lacks documentation",
                                    'file': str(file_path.relative_to(self.repo_root)),
                                    'line': line_num,
                                    'effort': 1,
                                    'priority': 'low',
                                    'source': 'documentation-analysis'
                                })
                    except (UnicodeDecodeError, IOError):
                        continue
        
        return opportunities
    
    def _find_performance_issues(self) -> List[Dict]:
        """Find potential performance issues"""
        opportunities = []
        
        performance_patterns = [
            (r'\.append\([^)]+\)\s*for\s+', 'Consider list comprehension instead of append in loop'),
            (r'len\([^)]+\)\s*==\s*0', 'Use "not container" instead of "len(container) == 0"'),
            (r'\.keys\(\)\s*in\s+', 'Use "in dict" instead of "in dict.keys()"'),
            (r'range\(len\([^)]+\)\)', 'Consider enumerate() instead of range(len())'),
        ]
        
        for root, dirs, files in os.walk(self.repo_root / "src"):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        for pattern, suggestion in performance_patterns:
                            for match in re.finditer(pattern, content):
                                line_num = content[:match.start()].count('\n') + 1
                                opportunities.append({
                                    'id': f"perf-{hash(f'{file_path}:{line_num}')}",
                                    'title': f"Performance optimization in {file}",
                                    'type': 'performance',
                                    'description': suggestion,
                                    'file': str(file_path.relative_to(self.repo_root)),
                                    'line': line_num,
                                    'effort': 1,
                                    'priority': 'medium',
                                    'source': 'performance-analysis'
                                })
                    except (UnicodeDecodeError, IOError):
                        continue
        
        return opportunities
    
    def _find_missing_tests(self) -> List[Dict]:
        """Find modules without corresponding test files"""
        opportunities = []
        
        src_files = set()
        test_files = set()
        
        # Collect source files
        for root, dirs, files in os.walk(self.repo_root / "src"):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    src_files.add(file[:-3])  # Remove .py extension
        
        # Collect test files
        if (self.repo_root / "tests").exists():
            for root, dirs, files in os.walk(self.repo_root / "tests"):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.add(file[5:-3])  # Remove test_ prefix and .py extension
        
        # Find missing tests
        missing_tests = src_files - test_files
        for module in list(missing_tests)[:5]:  # Limit to 5 missing tests
            opportunities.append({
                'id': f"test-{module}",
                'title': f"Add tests for {module} module",
                'type': 'testing',
                'description': f"Module {module} lacks test coverage",
                'effort': 3,
                'priority': 'medium',
                'source': 'test-analysis'
            })
        
        return opportunities
    
    def _estimate_comment_effort(self, comment: str) -> int:
        """Estimate effort based on comment content"""
        comment_lower = comment.lower()
        if any(word in comment_lower for word in ['complex', 'refactor', 'rewrite', 'redesign']):
            return 5
        elif any(word in comment_lower for word in ['fix', 'update', 'change', 'improve']):
            return 2
        else:
            return 1
    
    def _get_comment_priority(self, comment_type: str) -> str:
        """Get priority based on comment type"""
        priority_map = {
            'FIXME': 'high',
            'HACK': 'high', 
            'XXX': 'medium',
            'TODO': 'low'
        }
        return priority_map.get(comment_type, 'low')
    
    def _score_opportunity(self, opportunity: Dict) -> Dict:
        """Score opportunity using simplified algorithm"""
        # Base score from type
        type_scores = {
            'security': 100,
            'performance': 80,
            'technical-debt': 60,
            'testing': 50,
            'code-quality': 40,
            'documentation': 30
        }
        
        base_score = type_scores.get(opportunity.get('type'), 30)
        
        # Priority multiplier
        priority_multipliers = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.7
        }
        
        priority_mult = priority_multipliers.get(opportunity.get('priority'), 1.0)
        
        # Effort divisor (higher effort = lower score)
        effort = max(opportunity.get('effort', 1), 1)
        
        # Calculate final score
        score = (base_score * priority_mult) / effort
        
        opportunity['score'] = round(score, 1)
        return opportunity
    
    def generate_backlog(self, opportunities: List[Dict]) -> None:
        """Generate BACKLOG.md file"""
        now = datetime.now()
        
        content = f"""# 📊 Autonomous Value Backlog

*Last Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}*

## 🎯 Next Best Value Item

"""
        
        if opportunities:
            best = opportunities[0]
            content += f"""**{best['title']}**
- **Score**: {best.get('score', 0)}
- **Type**: {best.get('type', 'unknown')}
- **Effort**: {best.get('effort', 1)} hours
- **Priority**: {best.get('priority', 'medium')}
- **File**: {best.get('file', 'N/A')}
- **Description**: {best.get('description', 'No description')}

"""
        else:
            content += "No high-value opportunities identified at this time.\n\n"

        content += """## 📋 Value Opportunities

| Rank | Title | Score | Type | Effort | Priority |
|------|-------|--------|------|---------|----------|
"""
        
        for i, opp in enumerate(opportunities[:15], 1):
            title = opp['title'][:60] + "..." if len(opp['title']) > 60 else opp['title']
            content += f"| {i} | {title} | {opp.get('score', 0)} | {opp.get('type', 'unknown')} | {opp.get('effort', 1)}h | {opp.get('priority', 'medium')} |\n"
        
        # Summary statistics
        if opportunities:
            content += f"""

## 📊 Summary

- **Total Opportunities**: {len(opportunities)}
- **Average Score**: {sum(opp.get('score', 0) for opp in opportunities) / len(opportunities):.1f}
- **High Priority**: {len([opp for opp in opportunities if opp.get('priority') == 'high'])}
- **Medium Priority**: {len([opp for opp in opportunities if opp.get('priority') == 'medium'])}
- **Low Priority**: {len([opp for opp in opportunities if opp.get('priority') == 'low'])}

### By Type
"""
            # Count by type
            type_counts = {}
            for opp in opportunities:
                opp_type = opp.get('type', 'unknown')
                type_counts[opp_type] = type_counts.get(opp_type, 0) + 1
            
            for opp_type, count in sorted(type_counts.items()):
                content += f"- **{opp_type.title()}**: {count}\n"
        
        content += f"""

---
*Generated by Terragon Autonomous SDLC Enhancement System*
*Repository Maturity Level: MATURING (50-75%)*
"""
        
        with open(self.backlog_file, 'w') as f:
            f.write(content)
    
    def save_metrics(self, opportunities: List[Dict]) -> None:
        """Save metrics to JSON file"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "repository_maturity": "maturing",
            "total_opportunities": len(opportunities),
            "average_score": sum(opp.get('score', 0) for opp in opportunities) / len(opportunities) if opportunities else 0,
            "top_opportunity": opportunities[0] if opportunities else None,
            "breakdown_by_type": {},
            "breakdown_by_priority": {},
            "breakdown_by_source": {}
        }
        
        # Calculate breakdowns
        for opp in opportunities:
            # Type breakdown
            opp_type = opp.get('type', 'unknown')
            metrics['breakdown_by_type'][opp_type] = metrics['breakdown_by_type'].get(opp_type, 0) + 1
            
            # Priority breakdown  
            priority = opp.get('priority', 'unknown')
            metrics['breakdown_by_priority'][priority] = metrics['breakdown_by_priority'].get(priority, 0) + 1
            
            # Source breakdown
            source = opp.get('source', 'unknown')
            metrics['breakdown_by_source'][source] = metrics['breakdown_by_source'].get(source, 0) + 1
        
        # Ensure directory exists
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    """Main execution function"""
    print("🚀 Starting Terragon Autonomous Value Discovery...")
    
    discovery = SimpleValueDiscovery()
    opportunities = discovery.discover_opportunities()
    
    print(f"📊 Discovered {len(opportunities)} value opportunities")
    
    if opportunities:
        print(f"🎯 Top opportunity: {opportunities[0]['title']} (Score: {opportunities[0].get('score', 0)})")
        
        # Generate outputs
        discovery.generate_backlog(opportunities)
        discovery.save_metrics(opportunities)
        
        print(f"📝 Generated BACKLOG.md with {len(opportunities)} opportunities")
        print("✅ Value discovery complete!")
        
        # Show top 3 opportunities
        print("\n🏆 Top 3 Value Opportunities:")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"{i}. {opp['title']} (Score: {opp.get('score', 0)}, Type: {opp.get('type', 'unknown')})")
    else:
        print("ℹ️  No value opportunities discovered at this time")
        # Still create empty backlog
        discovery.generate_backlog([])
        discovery.save_metrics([])

if __name__ == "__main__":
    main()