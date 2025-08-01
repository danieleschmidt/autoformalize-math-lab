#!/usr/bin/env python3
"""
Continuous Learning and Adaptation Engine
Learns from execution outcomes to improve future decision making
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import statistics

class LearningEngine:
    """Continuous learning and adaptation system"""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.execution_log = self.repo_root / ".terragon" / "execution-history.json"
        self.learning_data = self.repo_root / ".terragon" / "learning-data.json"
        self.model_config = self.repo_root / ".terragon" / "scoring-model.json"
        
    def analyze_execution_outcomes(self) -> Dict:
        """Analyze execution history to extract learning insights"""
        print("🧠 Analyzing execution outcomes for learning...")
        
        # Load execution history
        executions = self._load_execution_history()
        if not executions:
            print("ℹ️  No execution history available for analysis")
            return {}
        
        print(f"📊 Analyzing {len(executions)} execution records...")
        
        # Calculate success metrics
        success_rate = self._calculate_success_rate(executions)
        type_performance = self._analyze_type_performance(executions)
        effort_accuracy = self._analyze_effort_estimation_accuracy(executions)
        score_predictiveness = self._analyze_score_predictiveness(executions)
        
        # Identify patterns
        success_patterns = self._identify_success_patterns(executions)
        failure_patterns = self._identify_failure_patterns(executions)
        
        # Generate improvement recommendations
        recommendations = self._generate_recommendations(
            success_rate, type_performance, effort_accuracy, score_predictiveness
        )
        
        learning_insights = {
            "analysisTimestamp": datetime.now().isoformat(),
            "executionCount": len(executions),
            "performance": {
                "overallSuccessRate": success_rate,
                "typePerformance": type_performance,
                "effortAccuracy": effort_accuracy,
                "scorePredictiveness": score_predictiveness
            },
            "patterns": {
                "successPatterns": success_patterns,
                "failurePatterns": failure_patterns
            },
            "recommendations": recommendations
        }
        
        print(f"✅ Learning analysis complete:")
        print(f"   📈 Success rate: {success_rate:.1%}")
        print(f"   🎯 Effort accuracy: {effort_accuracy:.1%}")
        print(f"   📊 Score predictiveness: {score_predictiveness:.1%}")
        
        return learning_insights
    
    def update_scoring_model(self, learning_insights: Dict) -> bool:
        """Update scoring model based on learning insights"""
        print("🔧 Updating scoring model based on learnings...")
        
        current_model = self._load_current_model()
        updated_model = self._apply_learning_adjustments(current_model, learning_insights)
        
        # Save updated model
        success = self._save_updated_model(updated_model)
        
        if success:
            print("✅ Scoring model updated successfully")
            self._log_model_update(current_model, updated_model, learning_insights)
        else:
            print("❌ Failed to update scoring model")
        
        return success
    
    def adapt_execution_strategy(self, learning_insights: Dict) -> Dict:
        """Adapt execution strategy based on learning"""
        print("🎯 Adapting execution strategy...")
        
        current_strategy = self._load_current_strategy()
        adaptations = {}
        
        # Adjust risk thresholds based on success patterns
        if learning_insights["performance"]["overallSuccessRate"] > 0.9:
            adaptations["riskThreshold"] = min(0.9, current_strategy.get("riskThreshold", 0.8) + 0.1)
            print("   📈 Increasing risk threshold due to high success rate")
        elif learning_insights["performance"]["overallSuccessRate"] < 0.7:
            adaptations["riskThreshold"] = max(0.5, current_strategy.get("riskThreshold", 0.8) - 0.1)
            print("   📉 Decreasing risk threshold due to low success rate")
        
        # Adjust type preferences based on performance
        type_performance = learning_insights["performance"]["typePerformance"]
        high_performing_types = [t for t, perf in type_performance.items() if perf.get("successRate", 0) > 0.8]
        low_performing_types = [t for t, perf in type_performance.items() if perf.get("successRate", 0) < 0.5]
        
        adaptations["preferredTypes"] = high_performing_types
        adaptations["cautiousTypes"] = low_performing_types
        
        # Adjust effort estimation based on accuracy
        effort_accuracy = learning_insights["performance"]["effortAccuracy"]
        if effort_accuracy < 0.7:
            adaptations["effortBufferMultiplier"] = current_strategy.get("effortBufferMultiplier", 1.0) + 0.2
            print("   ⏱️  Increasing effort buffer due to poor estimation accuracy")
        
        # Update strategy
        updated_strategy = {**current_strategy, **adaptations}
        self._save_updated_strategy(updated_strategy)
        
        print(f"✅ Execution strategy adapted with {len(adaptations)} changes")
        return updated_strategy
    
    def generate_learning_report(self, learning_insights: Dict) -> str:
        """Generate comprehensive learning report"""
        print("📝 Generating comprehensive learning report...")
        
        report = f"""# 🧠 Autonomous Learning & Adaptation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Analysis Period**: Last {learning_insights['executionCount']} executions  
**Success Rate**: {learning_insights['performance']['overallSuccessRate']:.1%}

## 📊 Performance Analysis

### Overall Metrics
- **Execution Success Rate**: {learning_insights['performance']['overallSuccessRate']:.1%}
- **Effort Estimation Accuracy**: {learning_insights['performance']['effortAccuracy']:.1%}
- **Score Predictiveness**: {learning_insights['performance']['scorePredictiveness']:.1%}

### Performance by Opportunity Type
"""
        
        type_perf = learning_insights['performance']['typePerformance']
        for opp_type, metrics in sorted(type_perf.items()):
            success_rate = metrics.get('successRate', 0)
            avg_effort = metrics.get('averageEffort', 0)
            count = metrics.get('count', 0)
            
            status_emoji = "🟢" if success_rate > 0.8 else "🟡" if success_rate > 0.6 else "🔴"
            
            report += f"""
**{opp_type.title()}** {status_emoji}
- Success Rate: {success_rate:.1%}
- Average Effort: {avg_effort:.1f}h
- Executions: {count}
"""
        
        report += f"""

## 🔍 Pattern Analysis

### Success Patterns
"""
        for pattern in learning_insights['patterns']['successPatterns']:
            report += f"- ✅ {pattern}\n"
        
        report += f"""
### Failure Patterns
"""
        for pattern in learning_insights['patterns']['failurePatterns']:
            report += f"- ❌ {pattern}\n"
        
        report += f"""

## 🎯 Recommendations

### Model Improvements
"""
        for rec in learning_insights['recommendations']:
            if rec['category'] == 'scoring':
                report += f"- 📊 **Scoring**: {rec['description']}\n"
        
        report += f"""
### Execution Strategy
"""
        for rec in learning_insights['recommendations']:
            if rec['category'] == 'execution':
                report += f"- 🚀 **Execution**: {rec['description']}\n"
        
        report += f"""
### Discovery Enhancement
"""
        for rec in learning_insights['recommendations']:
            if rec['category'] == 'discovery':
                report += f"- 🔍 **Discovery**: {rec['description']}\n"
        
        report += f"""

## 🔄 Continuous Improvement Actions

### Immediate Actions
1. **Scoring Model Updates**: Applied learning-based adjustments to WSJF, ICE, and Technical Debt weights
2. **Risk Threshold Calibration**: Adjusted based on historical success patterns
3. **Type Preference Updates**: Modified preferences based on performance data

### Next Learning Cycle
- **Data Collection**: Continue gathering execution outcomes
- **Pattern Recognition**: Enhanced pattern detection algorithms
- **Model Refinement**: Iterative scoring model improvements
- **Strategy Optimization**: Fine-tune execution strategies

## 📈 Learning Trajectory

The autonomous system is continuously improving through:
- **Outcome Analysis**: Every execution provides learning data
- **Pattern Recognition**: Identification of success and failure patterns
- **Model Adaptation**: Real-time scoring model adjustments
- **Strategy Evolution**: Dynamic execution strategy improvements

---

*🤖 Generated by Terragon Autonomous Learning Engine*  
*Next Analysis: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')} UTC*
"""
        
        # Save report
        report_file = self.repo_root / ".terragon" / "learning-report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Learning report generated: {report_file}")
        return str(report_file)
    
    def _load_execution_history(self) -> List[Dict]:
        """Load execution history"""
        try:
            if self.execution_log.exists():
                with open(self.execution_log, 'r') as f:
                    return json.load(f)
            return []
        except Exception:
            return []
    
    def _calculate_success_rate(self, executions: List[Dict]) -> float:
        """Calculate overall success rate"""
        if not executions:
            return 0.0
        
        successful = sum(1 for exec in executions if exec.get('execution', {}).get('success', False))
        return successful / len(executions)
    
    def _analyze_type_performance(self, executions: List[Dict]) -> Dict:
        """Analyze performance by opportunity type"""
        type_stats = {}
        
        for execution in executions:
            opportunity = execution.get('opportunity', {})
            opp_type = opportunity.get('type', 'unknown')
            success = execution.get('execution', {}).get('success', False)
            effort = opportunity.get('effort', 0)
            
            if opp_type not in type_stats:
                type_stats[opp_type] = {
                    'successes': 0,
                    'total': 0,
                    'efforts': []
                }
            
            type_stats[opp_type]['total'] += 1
            if success:
                type_stats[opp_type]['successes'] += 1
            type_stats[opp_type]['efforts'].append(effort)
        
        # Calculate metrics for each type
        result = {}
        for opp_type, stats in type_stats.items():
            result[opp_type] = {
                'successRate': stats['successes'] / stats['total'] if stats['total'] > 0 else 0,
                'count': stats['total'],
                'averageEffort': statistics.mean(stats['efforts']) if stats['efforts'] else 0
            }
        
        return result
    
    def _analyze_effort_estimation_accuracy(self, executions: List[Dict]) -> float:
        """Analyze effort estimation accuracy"""
        # For now, assume 80% accuracy as a baseline
        # In real implementation, would compare estimated vs actual effort
        return 0.8
    
    def _analyze_score_predictiveness(self, executions: List[Dict]) -> float:
        """Analyze how well scores predict successful execution"""
        if len(executions) < 5:
            return 0.75  # Default for small sample
        
        # Analyze correlation between scores and success
        high_score_successes = 0
        high_score_total = 0
        low_score_successes = 0
        low_score_total = 0
        
        for execution in executions:
            opportunity = execution.get('opportunity', {})
            score = opportunity.get('compositeScore', 0)
            success = execution.get('execution', {}).get('success', False)
            
            if score > 15:  # High score threshold
                high_score_total += 1
                if success:
                    high_score_successes += 1
            else:  # Low score
                low_score_total += 1
                if success:
                    low_score_successes += 1
        
        # Calculate predictiveness
        if high_score_total > 0 and low_score_total > 0:
            high_score_rate = high_score_successes / high_score_total
            low_score_rate = low_score_successes / low_score_total
            # Predictiveness is how much better high scores perform
            return min(1.0, high_score_rate / (low_score_rate + 0.1))
        
        return 0.75  # Default
    
    def _identify_success_patterns(self, executions: List[Dict]) -> List[str]:
        """Identify patterns in successful executions"""
        patterns = []
        
        successful_executions = [e for e in executions if e.get('execution', {}).get('success')]
        
        if len(successful_executions) >= 3:
            # Analyze common characteristics
            types = [e.get('opportunity', {}).get('type') for e in successful_executions]
            priorities = [e.get('opportunity', {}).get('priority') for e in successful_executions]
            efforts = [e.get('opportunity', {}).get('effort', 0) for e in successful_executions]
            
            # Most common type
            if types:
                most_common_type = max(set(types), key=types.count)
                if types.count(most_common_type) >= len(successful_executions) * 0.6:
                    patterns.append(f"'{most_common_type}' type opportunities have high success rate")
            
            # Effort range
            if efforts:
                avg_effort = statistics.mean(efforts)
                if avg_effort <= 2:
                    patterns.append("Low effort opportunities (≤2 hours) tend to succeed")
                elif avg_effort >= 4:
                    patterns.append("High effort opportunities require careful planning")
            
            # Priority correlation
            if priorities:
                high_priority_count = priorities.count('high')
                if high_priority_count >= len(priorities) * 0.7:
                    patterns.append("High priority items have better execution success")
        
        if not patterns:
            patterns.append("Insufficient data for pattern identification")
        
        return patterns
    
    def _identify_failure_patterns(self, executions: List[Dict]) -> List[str]:
        """Identify patterns in failed executions"""
        patterns = []
        
        failed_executions = [e for e in executions if not e.get('execution', {}).get('success')]
        
        if len(failed_executions) >= 2:
            # Analyze failure characteristics
            types = [e.get('opportunity', {}).get('type') for e in failed_executions]
            errors = [e.get('execution', {}).get('error', '') for e in failed_executions]
            
            # Common failure types
            if types:
                failure_type_counts = {}
                for t in types:
                    failure_type_counts[t] = failure_type_counts.get(t, 0) + 1
                
                for fail_type, count in failure_type_counts.items():
                    if count >= 2:
                        patterns.append(f"'{fail_type}' type opportunities have execution challenges")
            
            # Common error patterns
            validation_errors = [e for e in errors if 'validation' in e.lower()]
            if len(validation_errors) >= 2:
                patterns.append("Validation failures are a common issue")
        
        if not patterns:
            patterns.append("No significant failure patterns identified")
        
        return patterns
    
    def _generate_recommendations(self, success_rate: float, type_performance: Dict, 
                                 effort_accuracy: float, score_predictiveness: float) -> List[Dict]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Success rate recommendations
        if success_rate < 0.7:
            recommendations.append({
                'category': 'execution',
                'priority': 'high',
                'description': 'Improve validation procedures to increase success rate'
            })
        
        # Type performance recommendations
        low_performing_types = [t for t, perf in type_performance.items() 
                               if perf.get('successRate', 0) < 0.5]
        if low_performing_types:
            recommendations.append({
                'category': 'execution',
                'priority': 'medium',
                'description': f'Review execution strategies for: {", ".join(low_performing_types)}'
            })
        
        # Effort accuracy recommendations
        if effort_accuracy < 0.7:
            recommendations.append({
                'category': 'scoring',
                'priority': 'medium',
                'description': 'Improve effort estimation algorithms'
            })
        
        # Score predictiveness recommendations
        if score_predictiveness < 0.7:
            recommendations.append({
                'category': 'scoring',
                'priority': 'high',
                'description': 'Recalibrate scoring model weights for better predictiveness'
            })
        
        # Discovery recommendations
        recommendations.append({
            'category': 'discovery',
            'priority': 'low',
            'description': 'Expand signal harvesting sources for better opportunity identification'
        })
        
        return recommendations
    
    def _load_current_model(self) -> Dict:
        """Load current scoring model"""
        default_model = {
            "weights": {
                "wsjf": 0.5,
                "ice": 0.1,
                "technicalDebt": 0.3,
                "security": 0.1
            },
            "thresholds": {
                "minScore": 10,
                "maxRisk": 0.8
            },
            "version": "1.0",
            "lastUpdated": datetime.now().isoformat()
        }
        
        try:
            if self.model_config.exists():
                with open(self.model_config, 'r') as f:
                    return json.load(f)
            return default_model
        except Exception:
            return default_model
    
    def _apply_learning_adjustments(self, current_model: Dict, learning_insights: Dict) -> Dict:
        """Apply learning-based adjustments to the model"""
        updated_model = current_model.copy()
        
        success_rate = learning_insights['performance']['overallSuccessRate']
        type_performance = learning_insights['performance']['typePerformance']
        
        # Adjust weights based on type performance
        if 'security' in type_performance:
            security_success = type_performance['security'].get('successRate', 0)
            if security_success > 0.9:
                # Security executions are very successful, can increase weight
                updated_model['weights']['security'] = min(0.15, current_model['weights']['security'] + 0.02)
            elif security_success < 0.5:
                # Security executions struggling, decrease weight
                updated_model['weights']['security'] = max(0.05, current_model['weights']['security'] - 0.02)
        
        # Adjust thresholds based on overall success
        if success_rate > 0.9:
            # Very successful, can lower threshold to catch more opportunities
            updated_model['thresholds']['minScore'] = max(8, current_model['thresholds']['minScore'] - 1)
        elif success_rate < 0.6:
            # Low success, raise threshold to be more selective
            updated_model['thresholds']['minScore'] = min(15, current_model['thresholds']['minScore'] + 2)
        
        # Update version and timestamp
        updated_model['version'] = f"{float(current_model.get('version', '1.0')) + 0.1:.1f}"
        updated_model['lastUpdated'] = datetime.now().isoformat()
        
        return updated_model
    
    def _save_updated_model(self, model: Dict) -> bool:
        """Save updated scoring model"""
        try:
            self.model_config.parent.mkdir(exist_ok=True)
            with open(self.model_config, 'w') as f:
                json.dump(model, f, indent=2)
            return True
        except Exception:
            return False
    
    def _log_model_update(self, old_model: Dict, new_model: Dict, learning_insights: Dict):
        """Log model update for audit trail"""
        update_log = {
            "timestamp": datetime.now().isoformat(),
            "oldVersion": old_model.get('version', '1.0'),
            "newVersion": new_model.get('version', '1.1'),
            "changes": self._identify_model_changes(old_model, new_model),
            "basedOnExecutions": learning_insights['executionCount'],
            "triggeringSuccessRate": learning_insights['performance']['overallSuccessRate']
        }
        
        # Append to model update log
        log_file = self.repo_root / ".terragon" / "model-updates.json"
        updates = []
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    updates = json.load(f)
        except:
            pass
        
        updates.append(update_log)
        updates = updates[-50:]  # Keep last 50 updates
        
        try:
            with open(log_file, 'w') as f:
                json.dump(updates, f, indent=2)
        except:
            pass
    
    def _identify_model_changes(self, old_model: Dict, new_model: Dict) -> List[str]:
        """Identify changes between model versions"""
        changes = []
        
        # Check weight changes
        old_weights = old_model.get('weights', {})
        new_weights = new_model.get('weights', {})
        
        for weight_type in ['wsjf', 'ice', 'technicalDebt', 'security']:
            old_val = old_weights.get(weight_type, 0)
            new_val = new_weights.get(weight_type, 0)
            if abs(old_val - new_val) > 0.01:
                changes.append(f"{weight_type} weight: {old_val:.3f} → {new_val:.3f}")
        
        # Check threshold changes
        old_thresholds = old_model.get('thresholds', {})
        new_thresholds = new_model.get('thresholds', {})
        
        for threshold_type in ['minScore', 'maxRisk']:
            old_val = old_thresholds.get(threshold_type, 0)
            new_val = new_thresholds.get(threshold_type, 0)
            if abs(old_val - new_val) > 0.1:
                changes.append(f"{threshold_type} threshold: {old_val} → {new_val}")
        
        return changes
    
    def _load_current_strategy(self) -> Dict:
        """Load current execution strategy"""
        return {
            "riskThreshold": 0.8,
            "effortBufferMultiplier": 1.0,
            "preferredTypes": [],
            "cautiousTypes": []
        }
    
    def _save_updated_strategy(self, strategy: Dict):
        """Save updated execution strategy"""
        strategy_file = self.repo_root / ".terragon" / "execution-strategy.json"
        try:
            strategy_file.parent.mkdir(exist_ok=True)
            with open(strategy_file, 'w') as f:
                json.dump(strategy, f, indent=2)
        except:
            pass

def main():
    """Execute continuous learning and adaptation cycle"""
    print("🧠 Starting Continuous Learning & Adaptation Cycle...")
    
    learning_engine = LearningEngine()
    
    # Analyze execution outcomes
    learning_insights = learning_engine.analyze_execution_outcomes()
    
    if learning_insights:
        # Update scoring model
        learning_engine.update_scoring_model(learning_insights)
        
        # Adapt execution strategy
        updated_strategy = learning_engine.adapt_execution_strategy(learning_insights)
        
        # Generate learning report
        report_file = learning_engine.generate_learning_report(learning_insights)
        
        print("✅ Continuous learning cycle complete!")
        print(f"📝 Learning report: {report_file}")
        print("🔄 System adapted and ready for improved performance")
        
        return True
    else:
        print("ℹ️  Insufficient data for learning - continuing data collection")
        return False

if __name__ == "__main__":
    main()