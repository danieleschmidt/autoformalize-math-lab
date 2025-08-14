"""Autonomous enhancement capabilities for mathematical formalization.

This module provides advanced autonomous features that learn and adapt
from formalization attempts, implementing self-improving algorithms.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .pipeline import FormalizationPipeline, FormalizationResult
from .self_correcting import SelfCorrectingPipeline
from ..utils.logging_config import setup_logger


class LearningMode(Enum):
    """Different learning strategies for autonomous enhancement."""
    PATTERN_RECOGNITION = "pattern_recognition"
    SUCCESS_AMPLIFICATION = "success_amplification"
    ERROR_PATTERN_AVOIDANCE = "error_pattern_avoidance"
    ADAPTIVE_PROMPTING = "adaptive_prompting"


@dataclass
class FormalizationPattern:
    """Represents a learned pattern from formalization attempts."""
    pattern_type: str
    latex_pattern: str
    formal_pattern: str
    success_rate: float
    usage_count: int = 0
    confidence_score: float = 0.0
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Metrics for autonomous learning performance."""
    patterns_learned: int = 0
    patterns_applied: int = 0
    improvement_rate: float = 0.0
    adaptation_cycles: int = 0
    learning_efficiency: float = 0.0


class AutonomousEnhancementEngine:
    """Advanced autonomous enhancement engine for mathematical formalization.
    
    This engine implements self-improving algorithms that learn from past
    formalization attempts and automatically adapt strategies.
    """
    
    def __init__(
        self,
        base_pipeline: FormalizationPipeline,
        learning_modes: List[LearningMode] = None,
        pattern_db_path: Optional[str] = None,
        min_confidence_threshold: float = 0.7,
        max_patterns: int = 1000
    ):
        """Initialize autonomous enhancement engine.
        
        Args:
            base_pipeline: Base formalization pipeline to enhance
            learning_modes: Learning strategies to enable
            pattern_db_path: Path to pattern database file
            min_confidence_threshold: Minimum confidence for pattern application
            max_patterns: Maximum number of patterns to store
        """
        self.base_pipeline = base_pipeline
        self.learning_modes = learning_modes or [
            LearningMode.PATTERN_RECOGNITION,
            LearningMode.SUCCESS_AMPLIFICATION,
            LearningMode.ADAPTIVE_PROMPTING
        ]
        self.min_confidence_threshold = min_confidence_threshold
        self.max_patterns = max_patterns
        
        self.logger = setup_logger(__name__)
        self.pattern_db_path = Path(pattern_db_path or "cache/autonomous_patterns.json")
        self.pattern_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Learning state
        self.learned_patterns: List[FormalizationPattern] = []
        self.learning_metrics = LearningMetrics()
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Load existing patterns
        self._load_pattern_database()
        
        self.logger.info(f"Autonomous enhancement engine initialized with {len(self.learned_patterns)} patterns")
    
    async def enhanced_formalize(
        self,
        latex_content: str,
        verify: bool = True,
        learn_from_result: bool = True,
        adapt_strategy: bool = True
    ) -> FormalizationResult:
        """Perform enhanced formalization with autonomous learning.
        
        Args:
            latex_content: LaTeX content to formalize
            verify: Whether to verify generated code
            learn_from_result: Whether to learn from this attempt
            adapt_strategy: Whether to adapt strategy based on patterns
            
        Returns:
            Enhanced FormalizationResult with learning metadata
        """
        start_time = time.time()
        
        # Pre-processing: Apply learned patterns
        enhanced_latex = await self._apply_preprocessing_patterns(latex_content)
        
        # Strategy adaptation based on learned patterns
        if adapt_strategy:
            adapted_config = await self._adapt_formalization_strategy(enhanced_latex)
            if adapted_config:
                # Create enhanced pipeline with adapted configuration
                enhanced_pipeline = self._create_enhanced_pipeline(adapted_config)
                result = await enhanced_pipeline.formalize(enhanced_latex, verify=verify)
            else:
                result = await self.base_pipeline.formalize(enhanced_latex, verify=verify)
        else:
            result = await self.base_pipeline.formalize(enhanced_latex, verify=verify)
        
        # Post-processing: Learn from the result
        if learn_from_result:
            await self._learn_from_result(latex_content, result)
        
        # Add autonomous enhancement metadata
        enhancement_time = time.time() - start_time
        if not hasattr(result, 'metadata') or result.metadata is None:
            result.metadata = {}
        
        result.metadata.update({
            'autonomous_enhancement': {
                'patterns_applied': self._count_applied_patterns(latex_content),
                'strategy_adapted': adapt_strategy,
                'learning_enabled': learn_from_result,
                'enhancement_time': enhancement_time,
                'total_patterns_available': len(self.learned_patterns)
            }
        })
        
        return result
    
    async def autonomous_batch_learning(
        self,
        training_data: List[Tuple[str, str]],
        validation_split: float = 0.2,
        learning_cycles: int = 3
    ) -> Dict[str, Any]:
        """Perform autonomous batch learning from training data.
        
        Args:
            training_data: List of (latex_content, expected_formal_code) pairs
            validation_split: Fraction of data to use for validation
            learning_cycles: Number of learning cycles to perform
            
        Returns:
            Learning results and performance metrics
        """
        self.logger.info(f"Starting autonomous batch learning with {len(training_data)} samples")
        
        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        learning_results = {
            'cycles': [],
            'final_metrics': {},
            'pattern_evolution': []
        }
        
        baseline_performance = await self._evaluate_performance(val_data)
        self.logger.info(f"Baseline performance: {baseline_performance['success_rate']:.2%}")
        
        for cycle in range(learning_cycles):
            cycle_start = time.time()
            patterns_before = len(self.learned_patterns)
            
            # Learning phase
            for latex_content, expected_code in train_data:
                result = await self.enhanced_formalize(
                    latex_content,
                    verify=True,
                    learn_from_result=True,
                    adapt_strategy=True
                )
                
                # Learn from expected vs actual code differences
                if result.success and result.formal_code:
                    await self._learn_from_comparison(
                        latex_content, 
                        result.formal_code, 
                        expected_code
                    )
            
            # Validation phase
            val_performance = await self._evaluate_performance(val_data)
            
            # Pattern optimization
            await self._optimize_patterns()
            
            cycle_time = time.time() - cycle_start
            patterns_learned = len(self.learned_patterns) - patterns_before
            
            cycle_results = {
                'cycle': cycle + 1,
                'patterns_learned': patterns_learned,
                'validation_performance': val_performance,
                'improvement_over_baseline': val_performance['success_rate'] - baseline_performance['success_rate'],
                'cycle_time': cycle_time
            }
            
            learning_results['cycles'].append(cycle_results)
            self.logger.info(f"Cycle {cycle + 1} completed: +{patterns_learned} patterns, "
                           f"{val_performance['success_rate']:.2%} success rate")
        
        # Final evaluation
        final_performance = await self._evaluate_performance(val_data)
        learning_results['final_metrics'] = {
            'total_patterns_learned': len(self.learned_patterns),
            'final_success_rate': final_performance['success_rate'],
            'total_improvement': final_performance['success_rate'] - baseline_performance['success_rate'],
            'learning_efficiency': self.learning_metrics.learning_efficiency
        }
        
        # Save learned patterns
        await self._save_pattern_database()
        
        self.logger.info(f"Autonomous learning completed. Final improvement: "
                        f"{learning_results['final_metrics']['total_improvement']:+.2%}")
        
        return learning_results
    
    async def _apply_preprocessing_patterns(self, latex_content: str) -> str:
        """Apply learned preprocessing patterns to enhance LaTeX content."""
        enhanced_content = latex_content
        
        for pattern in self.learned_patterns:
            if (pattern.pattern_type == "preprocessing" and 
                pattern.confidence_score >= self.min_confidence_threshold):
                
                # Apply pattern transformation
                if pattern.latex_pattern in enhanced_content:
                    enhanced_content = enhanced_content.replace(
                        pattern.latex_pattern, 
                        pattern.metadata.get('enhanced_latex', pattern.latex_pattern)
                    )
                    pattern.usage_count += 1
        
        return enhanced_content
    
    async def _adapt_formalization_strategy(self, latex_content: str) -> Optional[Dict[str, Any]]:
        """Adapt formalization strategy based on learned patterns."""
        relevant_patterns = []
        
        for pattern in self.learned_patterns:
            if (pattern.pattern_type == "strategy" and
                pattern.confidence_score >= self.min_confidence_threshold):
                
                # Check if pattern applies to current content
                if self._pattern_matches_content(pattern, latex_content):
                    relevant_patterns.append(pattern)
        
        if not relevant_patterns:
            return None
        
        # Select best pattern based on confidence and success rate
        best_pattern = max(relevant_patterns, 
                          key=lambda p: p.confidence_score * p.success_rate)
        
        return best_pattern.metadata.get('strategy_config', {})
    
    def _create_enhanced_pipeline(self, config: Dict[str, Any]) -> FormalizationPipeline:
        """Create enhanced pipeline with adapted configuration."""
        # For now, return the base pipeline
        # In a full implementation, this would create a customized pipeline
        return self.base_pipeline
    
    async def _learn_from_result(self, latex_content: str, result: FormalizationResult) -> None:
        """Learn patterns from formalization result."""
        if not result.success:
            await self._learn_failure_pattern(latex_content, result)
            return
        
        # Learn success patterns
        if LearningMode.SUCCESS_AMPLIFICATION in self.learning_modes:
            await self._learn_success_pattern(latex_content, result)
        
        # Learn error recovery patterns
        if result.correction_rounds > 0:
            await self._learn_correction_pattern(latex_content, result)
    
    async def _learn_success_pattern(self, latex_content: str, result: FormalizationResult) -> None:
        """Learn from successful formalization."""
        pattern = FormalizationPattern(
            pattern_type="success",
            latex_pattern=self._extract_latex_pattern(latex_content),
            formal_pattern=result.formal_code[:200] if result.formal_code else "",
            success_rate=1.0,
            confidence_score=0.8,
            metadata={
                'processing_time': result.processing_time,
                'verification_success': result.verification_status,
                'correction_rounds': result.correction_rounds
            }
        )
        
        self._add_pattern(pattern)
    
    async def _learn_failure_pattern(self, latex_content: str, result: FormalizationResult) -> None:
        """Learn from failed formalization."""
        if LearningMode.ERROR_PATTERN_AVOIDANCE in self.learning_modes:
            pattern = FormalizationPattern(
                pattern_type="failure_avoidance",
                latex_pattern=self._extract_latex_pattern(latex_content),
                formal_pattern="",
                success_rate=0.0,
                confidence_score=0.6,
                metadata={
                    'error_message': result.error_message,
                    'failure_type': self._classify_failure_type(result)
                }
            )
            
            self._add_pattern(pattern)
    
    async def _learn_correction_pattern(self, latex_content: str, result: FormalizationResult) -> None:
        """Learn from correction attempts."""
        if hasattr(result, 'metadata') and result.metadata:
            correction_history = result.metadata.get('correction_history', [])
            
            if correction_history:
                pattern = FormalizationPattern(
                    pattern_type="correction",
                    latex_pattern=self._extract_latex_pattern(latex_content),
                    formal_pattern=result.formal_code[:200] if result.formal_code else "",
                    success_rate=1.0 if result.verification_status else 0.0,
                    confidence_score=0.7,
                    metadata={
                        'correction_rounds': result.correction_rounds,
                        'correction_history': correction_history
                    }
                )
                
                self._add_pattern(pattern)
    
    async def _learn_from_comparison(
        self, 
        latex_content: str, 
        generated_code: str, 
        expected_code: str
    ) -> None:
        """Learn from comparison between generated and expected code."""
        similarity_score = self._compute_code_similarity(generated_code, expected_code)
        
        if similarity_score > 0.8:  # High similarity indicates good pattern
            pattern = FormalizationPattern(
                pattern_type="high_quality",
                latex_pattern=self._extract_latex_pattern(latex_content),
                formal_pattern=expected_code[:200],
                success_rate=similarity_score,
                confidence_score=similarity_score,
                metadata={
                    'similarity_score': similarity_score,
                    'generated_code': generated_code[:200],
                    'expected_code': expected_code[:200]
                }
            )
            
            self._add_pattern(pattern)
    
    def _extract_latex_pattern(self, latex_content: str) -> str:
        """Extract characteristic pattern from LaTeX content."""
        # Simplified pattern extraction - normalize and get key elements
        lines = latex_content.strip().split('\n')
        key_elements = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['theorem', 'lemma', 'definition', 'proof']):
                key_elements.append(line[:100])  # Limit length
        
        return ' | '.join(key_elements) if key_elements else latex_content[:100]
    
    def _pattern_matches_content(self, pattern: FormalizationPattern, content: str) -> bool:
        """Check if a pattern matches the given content."""
        # Simplified matching - in practice, this would use more sophisticated NLP
        pattern_words = set(pattern.latex_pattern.lower().split())
        content_words = set(content.lower().split())
        
        overlap = len(pattern_words.intersection(content_words))
        return overlap / max(len(pattern_words), 1) > 0.3
    
    def _classify_failure_type(self, result: FormalizationResult) -> str:
        """Classify the type of failure."""
        if not result.error_message:
            return "unknown"
        
        error_msg = result.error_message.lower()
        
        if "syntax" in error_msg:
            return "syntax_error"
        elif "type" in error_msg:
            return "type_error"
        elif "import" in error_msg or "module" in error_msg:
            return "import_error"
        elif "timeout" in error_msg:
            return "timeout"
        else:
            return "logic_error"
    
    def _compute_code_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity between two code snippets."""
        # Simplified similarity computation
        words1 = set(code1.lower().split())
        words2 = set(code2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _add_pattern(self, pattern: FormalizationPattern) -> None:
        """Add a pattern to the learned patterns database."""
        # Check for similar existing patterns
        for existing in self.learned_patterns:
            if (existing.pattern_type == pattern.pattern_type and
                self._compute_code_similarity(existing.latex_pattern, pattern.latex_pattern) > 0.8):
                
                # Update existing pattern
                existing.usage_count += 1
                existing.confidence_score = (existing.confidence_score + pattern.confidence_score) / 2
                existing.success_rate = (existing.success_rate + pattern.success_rate) / 2
                return
        
        # Add new pattern
        self.learned_patterns.append(pattern)
        self.learning_metrics.patterns_learned += 1
        
        # Limit pattern database size
        if len(self.learned_patterns) > self.max_patterns:
            # Remove lowest confidence patterns
            self.learned_patterns.sort(key=lambda p: p.confidence_score * p.usage_count)
            self.learned_patterns = self.learned_patterns[:-self.max_patterns//10]
    
    async def _optimize_patterns(self) -> None:
        """Optimize pattern database by removing low-quality patterns."""
        initial_count = len(self.learned_patterns)
        
        # Remove patterns with low confidence and low usage
        self.learned_patterns = [
            p for p in self.learned_patterns
            if p.confidence_score >= 0.3 or p.usage_count >= 2
        ]
        
        # Update confidence scores based on usage
        for pattern in self.learned_patterns:
            if pattern.usage_count > 0:
                usage_boost = min(0.2, pattern.usage_count * 0.05)
                pattern.confidence_score = min(1.0, pattern.confidence_score + usage_boost)
        
        removed_count = initial_count - len(self.learned_patterns)
        if removed_count > 0:
            self.logger.info(f"Optimized patterns: removed {removed_count} low-quality patterns")
    
    async def _evaluate_performance(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate performance on test data."""
        if not test_data:
            return {'success_rate': 0.0, 'average_time': 0.0}
        
        successful = 0
        total_time = 0.0
        
        for latex_content, expected_code in test_data[:10]:  # Limit for efficiency
            result = await self.enhanced_formalize(
                latex_content,
                verify=False,  # Skip verification for speed
                learn_from_result=False,
                adapt_strategy=True
            )
            
            if result.success:
                successful += 1
            total_time += result.processing_time
        
        evaluated_count = min(len(test_data), 10)
        return {
            'success_rate': successful / evaluated_count,
            'average_time': total_time / evaluated_count
        }
    
    def _count_applied_patterns(self, latex_content: str) -> int:
        """Count how many patterns were applied to the content."""
        count = 0
        for pattern in self.learned_patterns:
            if self._pattern_matches_content(pattern, latex_content):
                count += 1
        return count
    
    def _load_pattern_database(self) -> None:
        """Load patterns from database file."""
        try:
            if self.pattern_db_path.exists():
                with open(self.pattern_db_path, 'r') as f:
                    data = json.load(f)
                    
                self.learned_patterns = [
                    FormalizationPattern(**pattern_data)
                    for pattern_data in data.get('patterns', [])
                ]
                
                if 'metrics' in data:
                    metrics_data = data['metrics']
                    self.learning_metrics = LearningMetrics(**metrics_data)
                
                self.logger.info(f"Loaded {len(self.learned_patterns)} patterns from database")
        except Exception as e:
            self.logger.warning(f"Failed to load pattern database: {e}")
            self.learned_patterns = []
    
    async def _save_pattern_database(self) -> None:
        """Save patterns to database file."""
        try:
            data = {
                'patterns': [
                    {
                        'pattern_type': p.pattern_type,
                        'latex_pattern': p.latex_pattern,
                        'formal_pattern': p.formal_pattern,
                        'success_rate': p.success_rate,
                        'usage_count': p.usage_count,
                        'confidence_score': p.confidence_score,
                        'domain': p.domain,
                        'metadata': p.metadata
                    }
                    for p in self.learned_patterns
                ],
                'metrics': {
                    'patterns_learned': self.learning_metrics.patterns_learned,
                    'patterns_applied': self.learning_metrics.patterns_applied,
                    'improvement_rate': self.learning_metrics.improvement_rate,
                    'adaptation_cycles': self.learning_metrics.adaptation_cycles,
                    'learning_efficiency': self.learning_metrics.learning_efficiency
                }
            }
            
            with open(self.pattern_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved {len(self.learned_patterns)} patterns to database")
        except Exception as e:
            self.logger.error(f"Failed to save pattern database: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of autonomous learning progress."""
        return {
            'total_patterns': len(self.learned_patterns),
            'learning_modes': [mode.value for mode in self.learning_modes],
            'metrics': {
                'patterns_learned': self.learning_metrics.patterns_learned,
                'patterns_applied': self.learning_metrics.patterns_applied,
                'improvement_rate': self.learning_metrics.improvement_rate,
                'learning_efficiency': self.learning_metrics.learning_efficiency
            },
            'pattern_types': self._get_pattern_type_distribution(),
            'top_patterns': self._get_top_patterns(5)
        }
    
    def _get_pattern_type_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern types."""
        distribution = {}
        for pattern in self.learned_patterns:
            distribution[pattern.pattern_type] = distribution.get(pattern.pattern_type, 0) + 1
        return distribution
    
    def _get_top_patterns(self, n: int) -> List[Dict[str, Any]]:
        """Get top N patterns by confidence score."""
        sorted_patterns = sorted(
            self.learned_patterns,
            key=lambda p: p.confidence_score * p.usage_count,
            reverse=True
        )
        
        return [
            {
                'type': p.pattern_type,
                'confidence': p.confidence_score,
                'usage_count': p.usage_count,
                'success_rate': p.success_rate,
                'latex_preview': p.latex_pattern[:50] + "..." if len(p.latex_pattern) > 50 else p.latex_pattern
            }
            for p in sorted_patterns[:n]
        ]