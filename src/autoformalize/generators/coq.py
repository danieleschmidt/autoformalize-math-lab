"""Coq formal proof generator.

This module provides functionality to generate Coq formal proofs
from parsed mathematical content using LLM-based code generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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
class CoqGenerationResult:
    """Result of Coq code generation."""
    success: bool
    coq_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class CoqGenerator:
    """Generator for Coq formal proofs.
    
    This class uses large language models to convert parsed mathematical
    content into syntactically correct Coq code with proper imports,
    theorem statements, and proof tactics.
    """
    
    # Standard Coq imports for mathematical content
    STANDARD_IMPORTS = [
        "Require Import Arith.",
        "Require Import Logic.",
        "Require Import Reals.",
        "Require Import ZArith.",
        "Require Import QArith.",
        "Require Import Classical.",
        "Require Import Lia.",
    ]
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000
    ):
        """Initialize the Coq generator.
        
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
        """Generate Coq code from parsed mathematical content.
        
        Args:
            parsed_content: Parsed LaTeX mathematical content
            
        Returns:
            Generated Coq code as a string
        """
        try:
            self.logger.info(f"Generating Coq code using {self.model}")
            
            code_sections = []
            
            # Add imports
            imports = self._generate_imports(parsed_content)
            code_sections.append(imports)
            
            # Process definitions first
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
            
            self.logger.info("Coq code generation completed successfully")
            return full_code
            
        except Exception as e:
            self.logger.error(f"Coq generation failed: {e}")
            raise GenerationError(f"Failed to generate Coq code: {e}")
    
    def _generate_imports(self, parsed_content: ParsedContent) -> str:
        """Generate appropriate imports based on content analysis."""
        imports = set(self.STANDARD_IMPORTS)
        
        # Analyze content to determine additional imports needed
        all_text = ""
        for stmt in parsed_content.get_all_statements():
            all_text += stmt.statement + (stmt.proof or "")
        
        # Add domain-specific imports based on keywords
        keyword_imports = {
            "real": "Require Import Reals.",
            "finite": "Require Import Finite_sets.",
            "list": "Require Import Lists.List.",
            "set": "Require Import Sets.Ensembles.",
            "function": "Require Import Functions.",
            "relation": "Require Import Relations.",
            "order": "Require Import Orders.",
            "group": "Require Import Structures.Orders.",
            "field": "Require Import Setoids.Setoid.",
        }
        
        text_lower = all_text.lower()
        for keyword, import_stmt in keyword_imports.items():
            if keyword in text_lower:
                imports.add(import_stmt)
        
        return "\n".join(sorted(imports))
    
    async def _generate_definitions(self, definitions: List[MathematicalStatement]) -> str:
        """Generate Coq definitions."""
        self.logger.debug(f"Generating {len(definitions)} definitions")
        
        definitions_code = []
        
        for definition in definitions:
            try:
                coq_def = await self._generate_single_definition(definition)
                if coq_def:
                    definitions_code.append(coq_def)
            except Exception as e:
                self.logger.warning(f"Failed to generate definition '{definition.name}': {e}")
                # Add placeholder
                definitions_code.append(f"(* Definition: {definition.name or 'unnamed'} *)\n(* {definition.statement} *)\nDefinition placeholder_def : Prop := True.")
        
        return "\n\n".join(definitions_code)
    
    async def _generate_single_definition(self, definition: MathematicalStatement) -> Optional[str]:
        """Generate a single Coq definition."""
        prompt = self._create_definition_prompt(definition)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_coq_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for definition: {e}")
            return None
    
    async def _generate_theorems(self, theorems: List[MathematicalStatement]) -> str:
        """Generate Coq theorems."""
        self.logger.debug(f"Generating {len(theorems)} theorems")
        
        theorems_code = []
        
        for theorem in theorems:
            try:
                coq_theorem = await self._generate_single_theorem(theorem)
                if coq_theorem:
                    theorems_code.append(coq_theorem)
            except Exception as e:
                self.logger.warning(f"Failed to generate theorem '{theorem.name}': {e}")
                theorems_code.append(f"(* Theorem: {theorem.name or 'unnamed'} *)\n(* {theorem.statement} *)\nTheorem placeholder_theorem: True.\nProof.\n  trivial.\nQed.")
        
        return "\n\n".join(theorems_code)
    
    async def _generate_single_theorem(self, theorem: MathematicalStatement) -> Optional[str]:
        """Generate a single Coq theorem with proof."""
        prompt = self._create_theorem_prompt(theorem)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_coq_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for theorem: {e}")
            return None
    
    async def _generate_lemmas(self, lemmas: List[MathematicalStatement]) -> str:
        """Generate Coq lemmas."""
        self.logger.debug(f"Generating {len(lemmas)} lemmas")
        
        lemmas_code = []
        
        for lemma in lemmas:
            try:
                coq_lemma = await self._generate_single_lemma(lemma)
                if coq_lemma:
                    lemmas_code.append(coq_lemma)
            except Exception as e:
                self.logger.warning(f"Failed to generate lemma '{lemma.name}': {e}")
                lemmas_code.append(f"(* Lemma: {lemma.name or 'unnamed'} *)\n(* {lemma.statement} *)\nLemma placeholder_lemma: True.\nProof.\n  trivial.\nQed.")
        
        return "\n\n".join(lemmas_code)
    
    async def _generate_single_lemma(self, lemma: MathematicalStatement) -> Optional[str]:
        """Generate a single Coq lemma."""
        prompt = self._create_lemma_prompt(lemma)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_coq_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for lemma: {e}")
            return None
    
    async def _generate_statements(self, statements: List[MathematicalStatement], stmt_type: str) -> str:
        """Generate generic mathematical statements."""
        self.logger.debug(f"Generating {len(statements)} {stmt_type}s")
        
        statements_code = []
        
        for stmt in statements:
            try:
                coq_stmt = await self._generate_single_statement(stmt, stmt_type)
                if coq_stmt:
                    statements_code.append(coq_stmt)
            except Exception as e:
                self.logger.warning(f"Failed to generate {stmt_type} '{stmt.name}': {e}")
                statements_code.append(f"(* {stmt_type.title()}: {stmt.name or 'unnamed'} *)\n(* {stmt.statement} *)\n{stmt_type.title()} placeholder: True.\nProof.\n  trivial.\nQed.")
        
        return "\n\n".join(statements_code)
    
    async def _generate_single_statement(self, statement: MathematicalStatement, stmt_type: str) -> Optional[str]:
        """Generate a single Coq statement."""
        prompt = self._create_statement_prompt(statement, stmt_type)
        
        try:
            response = await self._call_llm(prompt)
            return self._extract_coq_code(response)
        except Exception as e:
            self.logger.error(f"LLM call failed for {stmt_type}: {e}")
            return None
    
    def _create_definition_prompt(self, definition: MathematicalStatement) -> str:
        """Create a prompt for generating a Coq definition."""
        return f"""Convert the following mathematical definition to Coq code.

Definition: {definition.name or 'unnamed_definition'}
Statement: {definition.statement}

Please provide:
1. A proper Coq definition with correct syntax
2. Use appropriate Coq types (Prop, Type, nat, Z, R, etc.)
3. Include proper type annotations
4. Add a brief comment explaining the definition

Generate only valid Coq code, no explanations outside of comments.
"""
    
    def _create_theorem_prompt(self, theorem: MathematicalStatement) -> str:
        """Create a prompt for generating a Coq theorem."""
        return f"""Convert the following mathematical theorem to Coq code with a complete proof.

Theorem: {theorem.name or 'unnamed_theorem'}
Statement: {theorem.statement}
Proof: {theorem.proof or 'Proof not provided in source'}

Please provide:
1. A proper Coq theorem statement with correct syntax
2. A complete proof using appropriate tactics (trivial, auto, lia, lra, induction, etc.)
3. Use Coq standard library lemmas when applicable
4. Include proper variable declarations and assumptions
5. Add comments to explain key proof steps

Generate only valid Coq code, no explanations outside of comments.
"""
    
    def _create_lemma_prompt(self, lemma: MathematicalStatement) -> str:
        """Create a prompt for generating a Coq lemma."""
        return f"""Convert the following mathematical lemma to Coq code.

Lemma: {lemma.name or 'unnamed_lemma'}
Statement: {lemma.statement}
Proof: {lemma.proof or 'Proof not provided'}

Please provide:
1. A proper Coq lemma statement
2. A complete proof using Coq tactics
3. Use Coq standard library when possible
4. Keep the proof concise but complete

Generate only valid Coq code, no explanations outside of comments.
"""
    
    def _create_statement_prompt(self, statement: MathematicalStatement, stmt_type: str) -> str:
        """Create a prompt for generating a generic Coq statement."""
        return f"""Convert the following mathematical {stmt_type} to Coq code.

{stmt_type.title()}: {statement.name or 'unnamed'}
Statement: {statement.statement}
Proof: {statement.proof or 'No proof provided'}

Generate valid Coq code with proper syntax and tactics.
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
                            "content": "You are an expert in Coq and formal mathematics. Generate only valid Coq code."
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
                    system="You are an expert in Coq and formal mathematics. Generate only valid Coq code.",
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
    
    def _extract_coq_code(self, response: str) -> str:
        """Extract Coq code from LLM response."""
        import re
        
        # Try to find Coq code blocks
        coq_blocks = re.findall(r'```(?:coq)?\n(.*?)```', response, re.DOTALL)
        if coq_blocks:
            return coq_blocks[0].strip()
        
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
