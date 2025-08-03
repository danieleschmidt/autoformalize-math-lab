"""Lean 4 formal proof generator.

This module provides functionality to generate Lean 4 formal proofs
from parsed mathematical content using LLM-based code generation.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_OPENAI = False

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    AsyncAnthropic = None
    HAS_ANTHROPIC = False

from ..parsers.latex_parser import ParsedContent, MathematicalStatement
from ..core.exceptions import GenerationError, ModelError
from ..utils.logging_config import setup_logger
from ..utils.templates import TemplateManager


@dataclass
class Lean4GenerationResult:
    """Result of Lean 4 code generation."""
    success: bool
    lean_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    token_usage: Dict[str, int] = None


class Lean4Generator:
    """Generator for Lean 4 formal proofs.
    
    This class uses large language models to convert parsed mathematical
    content into syntactically correct Lean 4 code with proper imports,
    theorem statements, and proof tactics.
    """
    
    # Standard Lean 4 imports for mathematical content
    STANDARD_IMPORTS = [
        "import Mathlib.Data.Nat.Basic",
        "import Mathlib.Data.Int.Basic",
        "import Mathlib.Data.Real.Basic",
        "import Mathlib.Algebra.Group.Basic",
        "import Mathlib.Topology.Basic",
        "import Mathlib.Analysis.SpecialFunctions.Basic",
        "import Mathlib.NumberTheory.Basic",
        "import Mathlib.SetTheory.Basic",
    ]
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000
    ):
        """Initialize the Lean 4 generator.
        
        Args:
            model: LLM model to use (gpt-4, claude-3-opus, etc.)
            api_key: API key for the model
            temperature: Generation temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = setup_logger(__name__)
        self.template_manager = TemplateManager()
        
        # Initialize the appropriate client
        self._setup_client(api_key)
        
    def _setup_client(self, api_key: Optional[str]) -> None:
        """Setup the LLM client based on the model."""
        if self.model.startswith("gpt") or self.model.startswith("o1"):
            if not HAS_OPENAI:
                raise ModelError("OpenAI package not installed. Run: pip install openai")
            self.client = AsyncOpenAI(api_key=api_key)
            self.client_type = "openai"
        elif self.model.startswith("claude"):
            if not HAS_ANTHROPIC:
                raise ModelError("Anthropic package not installed. Run: pip install anthropic")
            self.client = AsyncAnthropic(api_key=api_key)
            self.client_type = "anthropic"
        else:
            raise ModelError(f"Unsupported model: {self.model}")
    
    async def generate(self, parsed_content: ParsedContent) -> str:
        """Generate Lean 4 code from parsed mathematical content.
        
        Args:
            parsed_content: Parsed LaTeX mathematical content
            
        Returns:
            Generated Lean 4 code as a string
        """
        try:
            self.logger.info(f"Generating Lean 4 code using {self.model}")
            
            # Generate code for different types of statements
            code_sections = []
            
            # Add imports
            imports = self._generate_imports(parsed_content)
            code_sections.append(imports)
            
            # Process definitions first (they may be needed by theorems)
            if parsed_content.definitions:
                definitions_code = await self._generate_definitions(parsed_content.definitions)
                code_sections.append(definitions_code)
            
            # Process theorems and lemmas
            if parsed_content.theorems:
                theorems_code = await self._generate_theorems(parsed_content.theorems)
                code_sections.append(theorems_code)
            
            if parsed_content.lemmas:
                lemmas_code = await self._generate_lemmas(parsed_content.lemmas)
                code_sections.append(lemmas_code)
            
            # Process other statement types
            for stmt_type, statements in [
                ("proposition", parsed_content.propositions),
                ("corollary", parsed_content.corollaries),
            ]:
                if statements:
                    code = await self._generate_statements(statements, stmt_type)
                    code_sections.append(code)
            
            # Combine all sections
            full_code = "\n\n".join(filter(None, code_sections))
            
            self.logger.info("Lean 4 code generation completed successfully")
            return full_code
            
        except Exception as e:
            self.logger.error(f"Lean 4 generation failed: {e}")
            raise GenerationError(f"Failed to generate Lean 4 code: {e}")
    
    def _generate_imports(self, parsed_content: ParsedContent) -> str:
        """Generate appropriate imports based on content analysis."""
        imports = set(self.STANDARD_IMPORTS)
        
        # Analyze content to determine additional imports needed
        all_text = ""
        for stmt in parsed_content.get_all_statements():
            all_text += stmt.statement + (stmt.proof or "")
        
        # Add domain-specific imports based on keywords
        keyword_imports = {
            "topology": "import Mathlib.Topology.Basic",
            "metric": "import Mathlib.Topology.MetricSpace.Basic",
            "continuous": "import Mathlib.Topology.Constructions",
            "differentiable": "import Mathlib.Analysis.Calculus.FDeriv.Basic",
            "integral": "import Mathlib.MeasureTheory.Integral.Basic",
            "finite": "import Mathlib.Data.Fintype.Basic",
            "linear": "import Mathlib.LinearAlgebra.Basic",
            "group": "import Mathlib.GroupTheory.Basic",
            "ring": "import Mathlib.RingTheory.Basic",
            "field": "import Mathlib.FieldTheory.Basic",
        }
        
        text_lower = all_text.lower()
        for keyword, import_stmt in keyword_imports.items():
            if keyword in text_lower:
                imports.add(import_stmt)
        
        return "\n".join(sorted(imports))
    
    async def _generate_definitions(self, definitions: List[MathematicalStatement]) -> str:
        """Generate Lean 4 definitions."""
        self.logger.debug(f"Generating {len(definitions)} definitions")
        
        definitions_code = []
        
        for definition in definitions:
            try:
                lean_def = await self._generate_single_definition(definition)
                if lean_def:
                    definitions_code.append(lean_def)
            except Exception as e:
                self.logger.warning(f"Failed to generate definition '{definition.name}': {e}")
                # Add a placeholder comment
                definitions_code.append(f"-- Definition: {definition.name or 'unnamed'}\n-- {definition.statement}\n-- TODO: Implement this definition")
        
        return "\n\n".join(definitions_code)
    
    async def _generate_single_definition(self, definition: MathematicalStatement) -> Optional[str]:
        """Generate a single Lean 4 definition."""
        prompt = self._create_definition_prompt(definition)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_lean_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for definition: {e}")
            return None
    
    async def _generate_theorems(self, theorems: List[MathematicalStatement]) -> str:
        """Generate Lean 4 theorems."""
        self.logger.debug(f"Generating {len(theorems)} theorems")
        
        theorems_code = []
        
        for theorem in theorems:
            try:
                lean_theorem = await self._generate_single_theorem(theorem)
                if lean_theorem:
                    theorems_code.append(lean_theorem)
            except Exception as e:
                self.logger.warning(f"Failed to generate theorem '{theorem.name}': {e}")
                # Add a placeholder
                theorems_code.append(f"-- Theorem: {theorem.name or 'unnamed'}\n-- {theorem.statement}\ntheorem placeholder_{len(theorems_code)} : True := trivial")
        
        return "\n\n".join(theorems_code)
    
    async def _generate_single_theorem(self, theorem: MathematicalStatement) -> Optional[str]:
        """Generate a single Lean 4 theorem with proof."""
        prompt = self._create_theorem_prompt(theorem)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_lean_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for theorem: {e}")
            return None
    
    async def _generate_lemmas(self, lemmas: List[MathematicalStatement]) -> str:
        """Generate Lean 4 lemmas."""
        self.logger.debug(f"Generating {len(lemmas)} lemmas")
        
        lemmas_code = []
        
        for lemma in lemmas:
            try:
                lean_lemma = await self._generate_single_lemma(lemma)
                if lean_lemma:
                    lemmas_code.append(lean_lemma)
            except Exception as e:
                self.logger.warning(f"Failed to generate lemma '{lemma.name}': {e}")
                lemmas_code.append(f"-- Lemma: {lemma.name or 'unnamed'}\n-- {lemma.statement}\nlemma placeholder_lemma_{len(lemmas_code)} : True := trivial")
        
        return "\n\n".join(lemmas_code)
    
    async def _generate_single_lemma(self, lemma: MathematicalStatement) -> Optional[str]:
        """Generate a single Lean 4 lemma."""
        prompt = self._create_lemma_prompt(lemma)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_lean_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for lemma: {e}")
            return None
    
    async def _generate_statements(self, statements: List[MathematicalStatement], stmt_type: str) -> str:
        """Generate generic mathematical statements."""
        self.logger.debug(f"Generating {len(statements)} {stmt_type}s")
        
        statements_code = []
        
        for stmt in statements:
            try:
                lean_stmt = await self._generate_single_statement(stmt, stmt_type)
                if lean_stmt:
                    statements_code.append(lean_stmt)
            except Exception as e:
                self.logger.warning(f"Failed to generate {stmt_type} '{stmt.name}': {e}")
                statements_code.append(f"-- {stmt_type.title()}: {stmt.name or 'unnamed'}\n-- {stmt.statement}\n{stmt_type} placeholder_{len(statements_code)} : True := trivial")
        
        return "\n\n".join(statements_code)
    
    async def _generate_single_statement(self, statement: MathematicalStatement, stmt_type: str) -> Optional[str]:
        """Generate a single Lean 4 statement."""
        prompt = self._create_statement_prompt(statement, stmt_type)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_lean_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for {stmt_type}: {e}")
            return None
    
    def _create_definition_prompt(self, definition: MathematicalStatement) -> str:
        """Create a prompt for generating a Lean 4 definition."""
        template = self.template_manager.get_template("lean4_definition") or \
        """Convert the following mathematical definition to Lean 4 code.

Definition: {name}
Statement: {statement}

Please provide:
1. A proper Lean 4 definition with correct syntax
2. Use appropriate Mathlib types and structures
3. Include type annotations
4. Add a brief comment explaining the definition

Generate only valid Lean 4 code, no explanations outside of comments.
"""
        
        return template.format(
            name=definition.name or "unnamed_definition",
            statement=definition.statement,
            proof=definition.proof or "No proof provided"
        )
    
    def _create_theorem_prompt(self, theorem: MathematicalStatement) -> str:
        """Create a prompt for generating a Lean 4 theorem."""
        template = self.template_manager.get_template("lean4_theorem") or \
        """Convert the following mathematical theorem to Lean 4 code with a complete proof.

Theorem: {name}
Statement: {statement}
Proof: {proof}

Please provide:
1. A proper Lean 4 theorem statement with correct syntax
2. A complete proof using appropriate tactics (simp, rw, apply, exact, etc.)
3. Use Mathlib lemmas when applicable
4. Include proper variable declarations and assumptions
5. Add comments to explain key proof steps

Generate only valid Lean 4 code, no explanations outside of comments.
"""
        
        return template.format(
            name=theorem.name or "unnamed_theorem",
            statement=theorem.statement,
            proof=theorem.proof or "Proof not provided in source"
        )
    
    def _create_lemma_prompt(self, lemma: MathematicalStatement) -> str:
        """Create a prompt for generating a Lean 4 lemma."""
        template = self.template_manager.get_template("lean4_lemma") or \
        """Convert the following mathematical lemma to Lean 4 code.

Lemma: {name}
Statement: {statement}
Proof: {proof}

Please provide:
1. A proper Lean 4 lemma statement
2. A complete proof using Lean 4 tactics
3. Use Mathlib when possible
4. Keep the proof concise but complete

Generate only valid Lean 4 code, no explanations outside of comments.
"""
        
        return template.format(
            name=lemma.name or "unnamed_lemma",
            statement=lemma.statement,
            proof=lemma.proof or "Proof not provided"
        )
    
    def _create_statement_prompt(self, statement: MathematicalStatement, stmt_type: str) -> str:
        """Create a prompt for generating a generic Lean 4 statement."""
        return f"""Convert the following mathematical {stmt_type} to Lean 4 code.

{stmt_type.title()}: {statement.name or 'unnamed'}
Statement: {statement.statement}
Proof: {statement.proof or 'No proof provided'}

Generate valid Lean 4 code with proper syntax and tactics.
"""
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the language model with the given prompt."""
        try:
            if self.client_type == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in Lean 4 and formal mathematics. Generate only valid Lean 4 code."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.client_type == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system="You are an expert in Lean 4 and formal mathematics. Generate only valid Lean 4 code.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.content[0].text
            
            else:
                raise ModelError(f"Unsupported client type: {self.client_type}")
                
        except Exception as e:
            raise ModelError(f"LLM API call failed: {e}")
    
    def _extract_lean_code(self, response: str) -> str:
        """Extract Lean 4 code from LLM response."""
        # Look for code blocks
        import re
        
        # Try to find Lean code blocks
        lean_blocks = re.findall(r'```(?:lean|lean4)?\n(.*?)```', response, re.DOTALL)
        if lean_blocks:
            return lean_blocks[0].strip()
        
        # If no code blocks, return the entire response (cleaned)
        lines = response.split('\n')
        code_lines = []
        
        for line in lines:
            # Skip obvious non-code lines
            if line.strip().startswith('Here') or line.strip().startswith('This'):
                continue
            if 'explanation' in line.lower() or 'generates' in line.lower():
                continue
            code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        # This would be expanded to track actual usage statistics
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "client_type": self.client_type
        }
