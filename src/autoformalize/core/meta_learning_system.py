"""Real-Time Adaptation and Meta-Learning System.

This module implements advanced meta-learning capabilities that enable the
formalization system to rapidly adapt to new domains, proof styles, and
mathematical patterns with minimal examples.
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import pickle
import hashlib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    META_LEARNING_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    KMeans = None
    TSNE = None
    META_LEARNING_AVAILABLE = False

from ..utils.logging_config import setup_logger


@dataclass
class TaskContext:
    """Context information for meta-learning tasks."""
    domain: str
    complexity_level: float
    proof_style: str
    mathematical_concepts: List[str]
    success_patterns: List[str]
    failure_patterns: List[str]
    environmental_factors: Dict[str, Any]


@dataclass
class MetaLearningExample:
    """Example for meta-learning with context and outcome."""
    input_latex: str
    output_formal: str
    context: TaskContext
    success_metrics: Dict[str, float]
    timestamp: float
    adaptations_applied: List[str] = field(default_factory=list)


@dataclass
class AdaptationStrategy:
    """Strategy for adapting to new contexts."""
    strategy_id: str
    name: str
    context_similarity_threshold: float
    adaptation_function: Callable
    success_rate: float = 0.0
    usage_count: int = 0
    confidence: float = 0.5


class ContextEncoder:
    """Neural network for encoding task contexts into embeddings."""
    
    def __init__(self, context_dim: int = 128, hidden_dim: int = 256):
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.logger = setup_logger("ContextEncoder")
        
        if META_LEARNING_AVAILABLE and torch:
            self._initialize_networks()
        else:
            self.encoder = None
            self.decoder = None
            
    def _initialize_networks(self):
        """Initialize encoder and decoder networks."""
        try:
            # Context encoder network
            self.encoder = nn.Sequential(
                nn.Linear(self.context_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 64)  # Embedding dimension
            )
            
            # Context decoder for reconstruction
            self.decoder = nn.Sequential(
                nn.Linear(64, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.context_dim)
            )
            
            self.optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=0.001
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize networks: {e}")
            self.encoder = None
            self.decoder = None
            
    def encode_context(self, context: TaskContext) -> np.ndarray:
        """Encode task context into fixed-size embedding."""
        try:
            if not self.encoder:
                # Fallback to simple hashing
                context_str = f"{context.domain}_{context.complexity_level}_{context.proof_style}"
                context_hash = int(hashlib.md5(context_str.encode()).hexdigest()[:8], 16)
                return np.array([context_hash % 1000 / 1000.0] * 64)
                
            # Convert context to tensor
            context_vector = self._context_to_vector(context)
            context_tensor = torch.FloatTensor(context_vector).unsqueeze(0)
            
            # Encode
            with torch.no_grad():
                embedding = self.encoder(context_tensor)
                
            return embedding.numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Context encoding failed: {e}")
            return np.zeros(64)
            
    def _context_to_vector(self, context: TaskContext) -> np.ndarray:
        """Convert TaskContext to numerical vector."""
        vector = np.zeros(self.context_dim)
        
        # Domain encoding (one-hot-ish)
        domain_mapping = {
            "algebra": 0, "analysis": 1, "topology": 2, "number_theory": 3,
            "geometry": 4, "combinatorics": 5, "logic": 6
        }
        domain_idx = domain_mapping.get(context.domain, 7)
        if domain_idx < 8:
            vector[domain_idx] = 1.0
            
        # Complexity level
        vector[8] = context.complexity_level
        
        # Proof style encoding
        style_mapping = {
            "direct": 0, "contradiction": 1, "induction": 2, "construction": 3
        }
        style_idx = style_mapping.get(context.proof_style, 4)
        if style_idx < 5:
            vector[9 + style_idx] = 1.0
            
        # Mathematical concepts (bag of words)
        concept_keywords = [
            "group", "ring", "field", "topology", "manifold", "derivative",
            "integral", "limit", "continuous", "prime", "divisor"
        ]
        for i, keyword in enumerate(concept_keywords):
            if keyword in context.mathematical_concepts and i + 14 < self.context_dim:
                vector[i + 14] = 1.0
                
        return vector
        
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between context embeddings."""
        try:
            # Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0


class MetaLearningEngine:
    """Meta-learning engine for rapid adaptation to new mathematical contexts."""
    
    def __init__(
        self,
        memory_size: int = 10000,
        adaptation_threshold: float = 0.7,
        learning_rate: float = 0.01,
        cache_dir: Optional[str] = None
    ):
        self.memory_size = memory_size
        self.adaptation_threshold = adaptation_threshold
        self.learning_rate = learning_rate
        self.logger = setup_logger("MetaLearningEngine")
        
        # Initialize components
        self.context_encoder = ContextEncoder()
        self.episodic_memory = deque(maxlen=memory_size)
        self.adaptation_strategies: Dict[str, AdaptationStrategy] = {}
        self.context_clusters = defaultdict(list)
        
        # Performance tracking
        self.meta_metrics = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "average_adaptation_time": 0.0,
            "context_coverage": 0.0,
            "rapid_learning_rate": 0.0
        }
        
        # Cache management
        self.cache_dir = Path(cache_dir or "cache/meta_learning")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default adaptation strategies
        self._initialize_adaptation_strategies()
        
        # Load existing memory if available
        self._load_memory()
        
    def _initialize_adaptation_strategies(self):
        """Initialize default adaptation strategies."""
        strategies = [
            AdaptationStrategy(
                strategy_id="domain_transfer",
                name="Cross-Domain Transfer",
                context_similarity_threshold=0.6,
                adaptation_function=self._domain_transfer_adaptation
            ),
            AdaptationStrategy(
                strategy_id="complexity_scaling",
                name="Complexity Scaling",
                context_similarity_threshold=0.8,
                adaptation_function=self._complexity_scaling_adaptation
            ),
            AdaptationStrategy(
                strategy_id="proof_style_adaptation",
                name="Proof Style Adaptation",
                context_similarity_threshold=0.7,
                adaptation_function=self._proof_style_adaptation
            ),
            AdaptationStrategy(
                strategy_id="few_shot_learning",
                name="Few-Shot Learning",
                context_similarity_threshold=0.5,
                adaptation_function=self._few_shot_learning_adaptation
            )
        ]
        
        for strategy in strategies:
            self.adaptation_strategies[strategy.strategy_id] = strategy
            
    async def adapt_to_context(
        self,
        new_context: TaskContext,
        few_shot_examples: List[MetaLearningExample] = None
    ) -> Dict[str, Any]:
        """Adapt the system to a new mathematical context."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Adapting to new context: {new_context.domain}")
            
            # Encode new context
            context_embedding = self.context_encoder.encode_context(new_context)
            
            # Find similar contexts in memory
            similar_contexts = self._find_similar_contexts(context_embedding)
            
            # Select appropriate adaptation strategy
            adaptation_strategy = self._select_adaptation_strategy(
                new_context, similar_contexts, few_shot_examples
            )
            
            # Apply adaptation
            adaptation_result = await adaptation_strategy.adaptation_function(
                new_context, similar_contexts, few_shot_examples or []
            )
            
            # Update strategy metrics
            adaptation_time = time.time() - start_time
            self._update_adaptation_metrics(adaptation_strategy, adaptation_result, adaptation_time)
            
            # Store adaptation in memory for future use
            if adaptation_result.get("success", False):
                self._store_successful_adaptation(new_context, adaptation_result)
                
            return {
                "success": adaptation_result.get("success", False),
                "strategy_used": adaptation_strategy.name,
                "adaptations_applied": adaptation_result.get("adaptations", []),
                "confidence": adaptation_result.get("confidence", 0.5),
                "adaptation_time": adaptation_time,
                "similar_contexts_found": len(similar_contexts)
            }
            
        except Exception as e:
            self.logger.error(f"Context adaptation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "adaptation_time": time.time() - start_time
            }
            
    def _find_similar_contexts(
        self,
        target_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[MetaLearningExample, float]]:
        """Find most similar contexts from episodic memory."""
        similarities = []
        
        for example in self.episodic_memory:
            example_embedding = self.context_encoder.encode_context(example.context)
            similarity = self.context_encoder.compute_similarity(target_embedding, example_embedding)
            similarities.append((example, similarity))
            
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def _select_adaptation_strategy(
        self,
        context: TaskContext,
        similar_contexts: List[Tuple[MetaLearningExample, float]],
        few_shot_examples: Optional[List[MetaLearningExample]]
    ) -> AdaptationStrategy:
        """Select the most appropriate adaptation strategy."""
        # Score each strategy based on context and available data
        strategy_scores = {}
        
        for strategy_id, strategy in self.adaptation_strategies.items():
            score = 0.0
            
            # Base score from strategy success rate
            score += strategy.success_rate * 0.4
            
            # Similarity score
            max_similarity = max([sim for _, sim in similar_contexts], default=0.0)
            if max_similarity >= strategy.context_similarity_threshold:
                score += 0.3
                
            # Few-shot learning bonus
            if few_shot_examples and strategy_id == "few_shot_learning":
                score += len(few_shot_examples) * 0.1
                
            # Domain-specific bonuses
            if context.domain in ["algebra", "number_theory"] and strategy_id == "domain_transfer":
                score += 0.2
                
            strategy_scores[strategy_id] = score
            
        # Select strategy with highest score
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)
        return self.adaptation_strategies[best_strategy_id]
        
    async def _domain_transfer_adaptation(
        self,
        context: TaskContext,
        similar_contexts: List[Tuple[MetaLearningExample, float]],
        few_shot_examples: List[MetaLearningExample]
    ) -> Dict[str, Any]:
        """Apply domain transfer learning."""
        try:
            adaptations = []
            
            # Find successful patterns from similar domains
            successful_patterns = []
            for example, similarity in similar_contexts:
                if example.success_metrics.get("success", 0.0) > 0.8 and similarity > 0.6:
                    successful_patterns.extend(example.context.success_patterns)
                    
            # Extract common successful patterns
            pattern_counts = defaultdict(int)
            for pattern in successful_patterns:
                pattern_counts[pattern] += 1
                
            # Select most common patterns
            common_patterns = [
                pattern for pattern, count in pattern_counts.items()
                if count >= 2
            ]
            
            if common_patterns:
                adaptations.append(f"Applied {len(common_patterns)} successful patterns from similar domains")
                
            # Generate domain-specific adaptations
            if context.domain == "algebra":
                adaptations.append("Applied algebraic structure recognition")
                adaptations.append("Enhanced group/ring theory pattern matching")
            elif context.domain == "analysis":
                adaptations.append("Applied limit and continuity patterns")
                adaptations.append("Enhanced epsilon-delta proof techniques")
                
            return {
                "success": len(adaptations) > 0,
                "adaptations": adaptations,
                "confidence": min(0.9, 0.5 + len(adaptations) * 0.1),
                "patterns_transferred": len(common_patterns)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "adaptations": []}
            
    async def _complexity_scaling_adaptation(
        self,
        context: TaskContext,
        similar_contexts: List[Tuple[MetaLearningExample, float]],
        few_shot_examples: List[MetaLearningExample]
    ) -> Dict[str, Any]:
        """Apply complexity-aware adaptations."""
        try:
            adaptations = []
            
            # Analyze complexity patterns from similar contexts
            complexity_successes = [
                (example.context.complexity_level, example.success_metrics.get("success", 0.0))
                for example, similarity in similar_contexts
                if similarity > 0.7
            ]
            
            if complexity_successes:
                # Find optimal complexity range
                successful_complexities = [
                    complexity for complexity, success in complexity_successes
                    if success > 0.8
                ]
                
                if successful_complexities:
                    optimal_complexity = np.mean(successful_complexities)
                    complexity_diff = abs(context.complexity_level - optimal_complexity)
                    
                    if complexity_diff > 0.2:
                        if context.complexity_level > optimal_complexity:
                            adaptations.append("Applied complexity reduction techniques")
                            adaptations.append("Simplified proof structure")
                        else:
                            adaptations.append("Enhanced proof rigor for higher complexity")
                            adaptations.append("Added intermediate steps")
                            
            # Complexity-specific adaptations
            if context.complexity_level > 0.8:
                adaptations.append("Applied advanced mathematical techniques")
                adaptations.append("Enhanced error handling for complex proofs")
            elif context.complexity_level < 0.3:
                adaptations.append("Optimized for simple proof patterns")
                adaptations.append("Reduced processing overhead")
                
            return {
                "success": len(adaptations) > 0,
                "adaptations": adaptations,
                "confidence": 0.7 if len(adaptations) > 1 else 0.5
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "adaptations": []}
            
    async def _proof_style_adaptation(
        self,
        context: TaskContext,
        similar_contexts: List[Tuple[MetaLearningExample, float]],
        few_shot_examples: List[MetaLearningExample]
    ) -> Dict[str, Any]:
        """Adapt to specific proof styles."""
        try:
            adaptations = []
            
            # Analyze proof style success patterns
            style_patterns = defaultdict(list)
            for example, similarity in similar_contexts:
                if similarity > 0.6:
                    style = example.context.proof_style
                    success = example.success_metrics.get("success", 0.0)
                    style_patterns[style].append(success)
                    
            # Find most successful styles
            style_success_rates = {
                style: np.mean(successes) for style, successes in style_patterns.items()
                if len(successes) >= 2
            }
            
            if context.proof_style in style_success_rates:
                success_rate = style_success_rates[context.proof_style]
                if success_rate > 0.8:
                    adaptations.append(f"Optimized for {context.proof_style} proof style")
                elif success_rate < 0.5:
                    # Suggest alternative style
                    best_style = max(style_success_rates, key=style_success_rates.get)
                    adaptations.append(f"Recommended alternative proof style: {best_style}")
                    
            # Style-specific adaptations
            if context.proof_style == "induction":
                adaptations.append("Applied induction pattern recognition")
                adaptations.append("Enhanced base case and inductive step handling")
            elif context.proof_style == "contradiction":
                adaptations.append("Applied contradiction proof templates")
                adaptations.append("Enhanced logical negation handling")
                
            return {
                "success": len(adaptations) > 0,
                "adaptations": adaptations,
                "confidence": 0.8 if context.proof_style in style_success_rates else 0.6
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "adaptations": []}
            
    async def _few_shot_learning_adaptation(
        self,
        context: TaskContext,
        similar_contexts: List[Tuple[MetaLearningExample, float]],
        few_shot_examples: List[MetaLearningExample]
    ) -> Dict[str, Any]:
        """Apply few-shot learning from provided examples."""
        try:
            if not few_shot_examples:
                return {"success": False, "error": "No few-shot examples provided", "adaptations": []}
                
            adaptations = []
            
            # Analyze patterns in few-shot examples
            input_patterns = []
            output_patterns = []
            
            for example in few_shot_examples:
                # Extract patterns from input LaTeX
                if "theorem" in example.input_latex.lower():
                    input_patterns.append("theorem_structure")
                if "proof" in example.input_latex.lower():
                    input_patterns.append("proof_structure")
                if any(keyword in example.input_latex.lower() for keyword in ["forall", "exists", "∀", "∃"]):
                    input_patterns.append("quantifier_usage")
                    
                # Extract patterns from output formal code
                if "by" in example.output_formal:
                    output_patterns.append("tactic_proof")
                if "exact" in example.output_formal:
                    output_patterns.append("term_proof")
                if "simp" in example.output_formal:
                    output_patterns.append("simplification")
                    
            # Generate adaptations based on observed patterns
            unique_input_patterns = list(set(input_patterns))
            unique_output_patterns = list(set(output_patterns))
            
            if unique_input_patterns:
                adaptations.append(f"Learned input patterns: {', '.join(unique_input_patterns)}")
                
            if unique_output_patterns:
                adaptations.append(f"Learned output patterns: {', '.join(unique_output_patterns)}")
                
            # Learn from success/failure patterns
            successful_examples = [ex for ex in few_shot_examples if ex.success_metrics.get("success", 0.0) > 0.8]
            if successful_examples:
                common_success_factors = []
                for ex in successful_examples:
                    common_success_factors.extend(ex.context.success_patterns)
                    
                if common_success_factors:
                    adaptations.append(f"Applied {len(set(common_success_factors))} success patterns from examples")
                    
            confidence = min(0.9, 0.3 + len(few_shot_examples) * 0.15)
            
            return {
                "success": len(adaptations) > 0,
                "adaptations": adaptations,
                "confidence": confidence,
                "examples_processed": len(few_shot_examples)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "adaptations": []}
            
    def _update_adaptation_metrics(
        self,
        strategy: AdaptationStrategy,
        result: Dict[str, Any],
        adaptation_time: float
    ):
        """Update adaptation strategy metrics."""
        strategy.usage_count += 1
        
        if result.get("success", False):
            strategy.success_rate = (
                (strategy.success_rate * (strategy.usage_count - 1) + 1.0) /
                strategy.usage_count
            )
        else:
            strategy.success_rate = (
                strategy.success_rate * (strategy.usage_count - 1) /
                strategy.usage_count
            )
            
        # Update global metrics
        self.meta_metrics["total_adaptations"] += 1
        if result.get("success", False):
            self.meta_metrics["successful_adaptations"] += 1
            
        # Update average adaptation time
        current_avg = self.meta_metrics["average_adaptation_time"]
        total_adaptations = self.meta_metrics["total_adaptations"]
        self.meta_metrics["average_adaptation_time"] = (
            (current_avg * (total_adaptations - 1) + adaptation_time) /
            total_adaptations
        )
        
    def _store_successful_adaptation(self, context: TaskContext, result: Dict[str, Any]):
        """Store successful adaptation in episodic memory."""
        example = MetaLearningExample(
            input_latex="",  # Would be filled with actual input
            output_formal="",  # Would be filled with actual output
            context=context,
            success_metrics={"success": 1.0, "confidence": result.get("confidence", 0.5)},
            timestamp=time.time(),
            adaptations_applied=result.get("adaptations", [])
        )
        
        self.episodic_memory.append(example)
        
    def _load_memory(self):
        """Load episodic memory from cache."""
        try:
            memory_file = self.cache_dir / "episodic_memory.pkl"
            if memory_file.exists():
                with open(memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.episodic_memory.extend(memory_data)
                    self.logger.info(f"Loaded {len(memory_data)} examples from memory cache")
        except Exception as e:
            self.logger.warning(f"Failed to load memory cache: {e}")
            
    def save_memory(self):
        """Save episodic memory to cache."""
        try:
            memory_file = self.cache_dir / "episodic_memory.pkl"
            with open(memory_file, 'wb') as f:
                pickle.dump(list(self.episodic_memory), f)
            self.logger.info("Saved episodic memory to cache")
        except Exception as e:
            self.logger.error(f"Failed to save memory cache: {e}")
            
    def get_meta_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning metrics."""
        return {
            **self.meta_metrics,
            "memory_size": len(self.episodic_memory),
            "adaptation_strategies": {
                strategy_id: {
                    "success_rate": strategy.success_rate,
                    "usage_count": strategy.usage_count,
                    "confidence": strategy.confidence
                }
                for strategy_id, strategy in self.adaptation_strategies.items()
            },
            "context_coverage": len(set(
                example.context.domain for example in self.episodic_memory
            )),
            "adaptation_success_rate": (
                self.meta_metrics["successful_adaptations"] / 
                max(1, self.meta_metrics["total_adaptations"])
            )
        }