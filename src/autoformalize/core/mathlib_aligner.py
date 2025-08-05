"""Mathlib alignment for Lean 4 formalization.

This module provides functionality to align generated Lean 4 code with
the Mathlib library, finding relevant theorems and suggesting alignments.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..core.exceptions import FormalizationError
from ..utils.logging_config import setup_logger


@dataclass
class MathlibSuggestion:
    """Represents a Mathlib theorem suggestion."""
    theorem_name: str
    theorem_statement: str
    relevance_score: float
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    description: Optional[str] = None


@dataclass 
class AlignmentResult:
    """Result of Mathlib alignment."""
    success: bool
    aligned_code: Optional[str] = None
    suggestions: List[MathlibSuggestion] = None
    error_message: Optional[str] = None
    alignment_score: float = 0.0
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class MathlibAligner:
    """Aligner for integrating with Mathlib theorems and definitions.
    
    This class provides functionality to find relevant Mathlib theorems
    and align generated Lean 4 code with the standard library.
    """
    
    def __init__(
        self,
        mathlib_path: Optional[Path] = None,
        model: str = "gpt-4"
    ):
        """Initialize Mathlib aligner.
        
        Args:
            mathlib_path: Path to Mathlib installation
            model: LLM model for alignment suggestions
        """
        self.mathlib_path = mathlib_path
        self.model = model
        self.logger = setup_logger(__name__)
        
        # Cache for theorem database
        self._theorem_cache: Dict[str, List[MathlibSuggestion]] = {}
        
        # Common Mathlib theorem patterns
        self.common_theorems = {
            "addition": ["Nat.add_zero", "Nat.zero_add", "Nat.add_comm", "Nat.add_assoc"],
            "multiplication": ["Nat.mul_zero", "Nat.zero_mul", "Nat.mul_one", "Nat.one_mul"],
            "inequalities": ["Nat.le_refl", "Nat.le_trans", "Nat.lt_irrefl"],
            "sets": ["Set.mem_empty_iff", "Set.mem_univ", "Set.subset_def"],
            "functions": ["Function.comp_apply", "Function.id_comp", "Function.comp_id"],
            "groups": ["Group.mul_one", "Group.one_mul", "Group.mul_inv_cancel"],
            "topology": ["IsOpen.union", "IsOpen.inter", "continuous_id"],
            "analysis": ["continuous_add", "continuous_mul", "differentiable_add"]
        }
    
    async def find_similar_theorems(
        self,
        latex_statement: str,
        max_suggestions: int = 10
    ) -> List[MathlibSuggestion]:
        """Find Mathlib theorems similar to a LaTeX statement.
        
        Args:
            latex_statement: LaTeX mathematical statement
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of MathlibSuggestion objects
        """
        try:
            self.logger.debug(f"Finding similar theorems for: {latex_statement[:100]}...")
            
            # Extract mathematical concepts from LaTeX
            concepts = self._extract_concepts(latex_statement)
            
            # Find relevant theorem categories
            relevant_categories = self._categorize_concepts(concepts)
            
            # Generate suggestions
            suggestions = []
            
            for category in relevant_categories:
                category_theorems = self.common_theorems.get(category, [])
                for theorem_name in category_theorems[:3]:  # Limit per category
                    suggestion = MathlibSuggestion(
                        theorem_name=theorem_name,
                        theorem_statement=f"-- {theorem_name} (Mathlib theorem)",
                        relevance_score=self._calculate_relevance(concepts, theorem_name),
                        description=f"Standard Mathlib theorem for {category}"
                    )
                    suggestions.append(suggestion)
            
            # Sort by relevance score
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar theorems: {e}")
            return []
    
    async def generate_aligned_proof(
        self,
        latex_statement: str,
        use_theorems: Optional[List[str]] = None
    ) -> str:
        """Generate Lean 4 proof aligned with Mathlib.
        
        Args:
            latex_statement: LaTeX mathematical statement
            use_theorems: Specific theorems to use in the proof
            
        Returns:
            Generated Lean 4 proof using Mathlib
        """
        try:
            # Find relevant theorems if not specified
            if not use_theorems:
                suggestions = await self.find_similar_theorems(latex_statement)
                use_theorems = [s.theorem_name for s in suggestions[:3]]
            
            # Generate alignment prompt
            prompt = self._create_alignment_prompt(latex_statement, use_theorems)
            
            # This would use an LLM to generate the proof
            # For now, return a template
            return self._generate_proof_template(latex_statement, use_theorems)
            
        except Exception as e:
            self.logger.error(f"Failed to generate aligned proof: {e}")
            raise FormalizationError(f"Proof generation failed: {e}")
    
    async def align_existing_code(
        self,
        lean_code: str,
        improve_alignment: bool = True
    ) -> AlignmentResult:
        """Align existing Lean 4 code with Mathlib standards.
        
        Args:
            lean_code: Existing Lean 4 code
            improve_alignment: Whether to suggest improvements
            
        Returns:
            AlignmentResult with alignment suggestions
        """
        try:
            self.logger.debug("Aligning existing code with Mathlib")
            
            # Analyze current code
            analysis = self._analyze_code(lean_code)
            
            # Find missing imports
            missing_imports = self._find_missing_imports(lean_code)
            
            # Find theorem substitutions
            substitutions = self._find_theorem_substitutions(lean_code)
            
            # Generate improved code if requested
            aligned_code = lean_code
            if improve_alignment:
                aligned_code = self._apply_alignments(
                    lean_code, 
                    missing_imports, 
                    substitutions
                )
            
            # Create suggestions
            suggestions = []
            for theorem_name, replacement in substitutions.items():
                suggestion = MathlibSuggestion(
                    theorem_name=replacement,
                    theorem_statement=f"Use {replacement} instead of {theorem_name}",
                    relevance_score=0.8,
                    description="Standard Mathlib equivalent"
                )
                suggestions.append(suggestion)
            
            alignment_score = self._calculate_alignment_score(aligned_code)
            
            return AlignmentResult(
                success=True,
                aligned_code=aligned_code,
                suggestions=suggestions,
                alignment_score=alignment_score
            )
            
        except Exception as e:
            self.logger.error(f"Code alignment failed: {e}")
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )
    
    def _extract_concepts(self, latex_statement: str) -> List[str]:
        """Extract mathematical concepts from LaTeX statement.
        
        Args:
            latex_statement: LaTeX mathematical statement
            
        Returns:
            List of mathematical concepts
        """
        concepts = []
        
        # Pattern matching for common mathematical concepts
        concept_patterns = {
            "addition": [r"\+", r"sum", r"add"],
            "multiplication": [r"\*", r"\\cdot", r"product", r"mult"],
            "inequalities": [r"\\leq", r"\\geq", r"<", r">", r"\\le", r"\\ge"],
            "sets": [r"\\in", r"\\subset", r"\\cup", r"\\cap", r"set"],
            "functions": [r"f\(", r"\\to", r"function", r"map"],
            "groups": [r"group", r"\\cdot", r"identity", r"inverse"],
            "topology": [r"open", r"closed", r"continuous", r"compact"],
            "analysis": [r"derivative", r"integral", r"limit", r"convergent"]
        }
        
        latex_lower = latex_statement.lower()
        
        for concept, patterns in concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, latex_lower):
                    concepts.append(concept)
                    break
        
        return list(set(concepts))  # Remove duplicates
    
    def _categorize_concepts(self, concepts: List[str]) -> List[str]:
        """Categorize mathematical concepts for theorem matching.
        
        Args:
            concepts: List of mathematical concepts
            
        Returns:
            List of relevant theorem categories
        """
        # Direct mapping and some category expansions
        categories = set(concepts)
        
        # Add related categories
        if "addition" in concepts or "multiplication" in concepts:
            categories.add("arithmetic")
        
        if "inequalities" in concepts:
            categories.add("order")
            
        if "topology" in concepts or "analysis" in concepts:
            categories.add("continuous")
        
        return list(categories)
    
    def _calculate_relevance(self, concepts: List[str], theorem_name: str) -> float:
        """Calculate relevance score between concepts and theorem.
        
        Args:
            concepts: List of mathematical concepts
            theorem_name: Name of the theorem
            
        Returns:
            Relevance score (0-1)
        """
        theorem_lower = theorem_name.lower()
        
        # Base score
        score = 0.1
        
        # Exact concept matches
        for concept in concepts:
            if concept.lower() in theorem_lower:
                score += 0.3
        
        # Pattern matching
        if "add" in theorem_lower and "addition" in concepts:
            score += 0.2
        if "mul" in theorem_lower and "multiplication" in concepts:
            score += 0.2
        if ("le" in theorem_lower or "lt" in theorem_lower) and "inequalities" in concepts:
            score += 0.2
        
        return min(score, 1.0)
    
    def _create_alignment_prompt(self, latex_statement: str, theorems: List[str]) -> str:
        """Create prompt for generating aligned proof.
        
        Args:
            latex_statement: LaTeX mathematical statement
            theorems: List of theorems to use
            
        Returns:
            Formatted alignment prompt
        """
        theorem_list = "\n".join(f"- {theorem}" for theorem in theorems)
        
        return f"""Generate a Lean 4 proof for the following mathematical statement using Mathlib:

Statement: {latex_statement}

Use these Mathlib theorems where appropriate:
{theorem_list}

Requirements:
1. Include proper Mathlib imports
2. Use idiomatic Lean 4 syntax
3. Prefer Mathlib theorems over custom proofs
4. Include type annotations where helpful

Generate only valid Lean 4 code."""
    
    def _generate_proof_template(self, latex_statement: str, theorems: List[str]) -> str:
        """Generate a basic proof template using specified theorems.
        
        Args:
            latex_statement: LaTeX mathematical statement
            theorems: List of theorems to reference
            
        Returns:
            Basic Lean 4 proof template
        """
        imports = self._generate_imports(theorems)
        
        return f"""{imports}

-- Generated proof for: {latex_statement}
theorem generated_theorem : Sorry := by
  -- This proof would use the following Mathlib theorems:
{chr(10).join(f"  -- {theorem}" for theorem in theorems)}
  sorry
"""
    
    def _generate_imports(self, theorems: List[str]) -> str:
        """Generate appropriate imports for given theorems.
        
        Args:
            theorems: List of theorem names
            
        Returns:
            Import statements
        """
        import_map = {
            "Nat.": "import Mathlib.Data.Nat.Basic",
            "Int.": "import Mathlib.Data.Int.Basic", 
            "Real.": "import Mathlib.Data.Real.Basic",
            "Set.": "import Mathlib.Data.Set.Basic",
            "Function.": "import Mathlib.Logic.Function.Basic",
            "Group.": "import Mathlib.Algebra.Group.Basic",
            "continuous": "import Mathlib.Topology.Basic"
        }
        
        imports = set()
        for theorem in theorems:
            for prefix, import_stmt in import_map.items():
                if theorem.startswith(prefix) or prefix.lower() in theorem.lower():
                    imports.add(import_stmt)
        
        # Add default imports
        imports.add("import Mathlib.Tactic")
        
        return "\n".join(sorted(imports))
    
    def _analyze_code(self, lean_code: str) -> Dict[str, Any]:
        """Analyze Lean 4 code for alignment opportunities.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            Analysis results
        """
        return {
            "has_imports": "import" in lean_code,
            "uses_mathlib": "Mathlib" in lean_code,
            "theorem_count": len(re.findall(r"theorem\s+\w+", lean_code)),
            "lemma_count": len(re.findall(r"lemma\s+\w+", lean_code)),
            "definition_count": len(re.findall(r"def\s+\w+", lean_code))
        }
    
    def _find_missing_imports(self, lean_code: str) -> List[str]:
        """Find missing imports for the given code.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            List of suggested imports
        """
        missing = []
        
        # Check for common patterns that need imports
        if re.search(r"\bNat\.", lean_code) and "Mathlib.Data.Nat" not in lean_code:
            missing.append("import Mathlib.Data.Nat.Basic")
        
        if re.search(r"\bReal\.", lean_code) and "Mathlib.Data.Real" not in lean_code:
            missing.append("import Mathlib.Data.Real.Basic")
        
        if "simp" in lean_code and "Mathlib.Tactic" not in lean_code:
            missing.append("import Mathlib.Tactic")
        
        return missing
    
    def _find_theorem_substitutions(self, lean_code: str) -> Dict[str, str]:
        """Find opportunities to substitute custom proofs with Mathlib theorems.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            Dictionary mapping current code to Mathlib alternatives
        """
        substitutions = {}
        
        # Pattern-based substitutions
        if "n + 0 = n" in lean_code:
            substitutions["n + 0 = n"] = "Nat.add_zero n"
        
        if "0 + n = n" in lean_code:
            substitutions["0 + n = n"] = "Nat.zero_add n"
        
        return substitutions
    
    def _apply_alignments(
        self,
        lean_code: str,
        missing_imports: List[str],
        substitutions: Dict[str, str]
    ) -> str:
        """Apply alignment improvements to Lean code.
        
        Args:
            lean_code: Original Lean 4 code
            missing_imports: List of imports to add
            substitutions: Dictionary of code substitutions
            
        Returns:
            Improved Lean 4 code
        """
        improved_code = lean_code
        
        # Add missing imports at the top
        if missing_imports:
            import_block = "\n".join(missing_imports) + "\n\n"
            if improved_code.startswith("import"):
                # Find end of existing imports
                lines = improved_code.split("\n")
                import_end = 0
                for i, line in enumerate(lines):
                    if not line.strip().startswith("import") and line.strip():
                        import_end = i
                        break
                
                lines = lines[:import_end] + missing_imports + lines[import_end:]
                improved_code = "\n".join(lines)
            else:
                improved_code = import_block + improved_code
        
        # Apply substitutions
        for old_pattern, new_pattern in substitutions.items():
            improved_code = improved_code.replace(old_pattern, new_pattern)
        
        return improved_code
    
    def _calculate_alignment_score(self, lean_code: str) -> float:
        """Calculate alignment score for Lean code.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            Alignment score (0-1)
        """
        score = 0.0
        
        # Check for Mathlib usage
        if "Mathlib" in lean_code:
            score += 0.3
        
        # Check for proper imports
        if "import" in lean_code:
            score += 0.2
        
        # Check for standard theorem usage
        mathlib_patterns = ["Nat.", "Int.", "Real.", "Set.", "Function."]
        for pattern in mathlib_patterns:
            if pattern in lean_code:
                score += 0.1
        
        return min(score, 1.0)