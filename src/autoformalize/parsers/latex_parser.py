"""LaTeX mathematical content parser.

This module provides functionality to parse LaTeX documents and extract
mathematical content including theorems, definitions, proofs, and lemmas.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
import asyncio

try:
    from pylatexenc.latex2text import LatexNodes2Text
    from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexEnvironmentNode
    HAS_PYLATEXENC = True
except ImportError:
    HAS_PYLATEXENC = False
    LatexNodes2Text = None
    LatexWalker = None

from ..core.exceptions import ParseError
from ..utils.logging_config import setup_logger


@dataclass
class MathematicalStatement:
    """Represents a mathematical statement (theorem, lemma, definition, etc.)."""
    type: str  # theorem, lemma, definition, corollary, etc.
    name: Optional[str] = None
    label: Optional[str] = None
    statement: str = ""
    proof: Optional[str] = None
    line_number: int = 0
    dependencies: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"{self.type.title()}{f' ({self.name})' if self.name else ''}: {self.statement[:100]}..."


@dataclass
class ParsedContent:
    """Container for parsed LaTeX mathematical content."""
    theorems: List[MathematicalStatement] = field(default_factory=list)
    definitions: List[MathematicalStatement] = field(default_factory=list)
    lemmas: List[MathematicalStatement] = field(default_factory=list)
    corollaries: List[MathematicalStatement] = field(default_factory=list)
    propositions: List[MathematicalStatement] = field(default_factory=list)
    examples: List[MathematicalStatement] = field(default_factory=list)
    raw_math: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_statements(self) -> List[MathematicalStatement]:
        """Get all mathematical statements."""
        return (self.theorems + self.definitions + self.lemmas + 
                self.corollaries + self.propositions + self.examples)
    
    def count_statements(self) -> Dict[str, int]:
        """Count statements by type."""
        return {
            'theorems': len(self.theorems),
            'definitions': len(self.definitions),
            'lemmas': len(self.lemmas),
            'corollaries': len(self.corollaries),
            'propositions': len(self.propositions),
            'examples': len(self.examples),
            'total': len(self.get_all_statements())
        }


class LaTeXParser:
    """Parser for extracting mathematical content from LaTeX documents.
    
    This class provides methods to parse LaTeX source code and extract
    mathematical statements such as theorems, definitions, proofs, etc.
    
    The parser supports both regex-based parsing (fallback) and advanced
    parsing using pylatexenc for better accuracy.
    """
    
    # Common mathematical environments
    MATH_ENVIRONMENTS = {
        'theorem', 'lemma', 'corollary', 'proposition', 'definition',
        'example', 'remark', 'note', 'claim', 'fact', 'observation',
        'conjecture', 'hypothesis', 'axiom', 'postulate'
    }
    
    # Proof environments
    PROOF_ENVIRONMENTS = {'proof', 'solution', 'sketch'}
    
    def __init__(self, custom_environments: Optional[Set[str]] = None):
        """Initialize the LaTeX parser.
        
        Args:
            custom_environments: Additional mathematical environments to recognize
        """
        self.logger = setup_logger(__name__)
        self.custom_environments = custom_environments or set()
        self.all_environments = self.MATH_ENVIRONMENTS | self.custom_environments
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
    def _compile_patterns(self) -> None:
        """Compile regex patterns for parsing."""
        # Environment pattern
        env_names = '|'.join(self.all_environments)
        self.env_pattern = re.compile(
            r'\\begin\{(' + env_names + r')\}(?:\[([^\]]*)\])?(?:\{([^}]*)\})?(.*?)\\end\{\1\}',
            re.DOTALL | re.IGNORECASE
        )
        
        # Proof pattern
        proof_names = '|'.join(self.PROOF_ENVIRONMENTS)
        self.proof_pattern = re.compile(
            r'\\begin\{(' + proof_names + r')\}(.*?)\\end\{\1\}',
            re.DOTALL | re.IGNORECASE
        )
        
        # Label pattern
        self.label_pattern = re.compile(r'\\label\{([^}]+)\}')
        
        # Math mode patterns
        self.inline_math_pattern = re.compile(r'\$(.*?)\$', re.DOTALL)
        self.display_math_pattern = re.compile(r'\$\$(.*?)\$\$|\\\[(.*?)\\\]', re.DOTALL)
        
        # Section patterns for context
        self.section_pattern = re.compile(
            r'\\(sub)*(section|chapter)\*?\{([^}]+)\}',
            re.IGNORECASE
        )
    
    async def parse(self, latex_content: str) -> ParsedContent:
        """Parse LaTeX content and extract mathematical statements.
        
        Args:
            latex_content: LaTeX source code as string
            
        Returns:
            ParsedContent object with extracted mathematical content
        """
        try:
            self.logger.info("Starting LaTeX parsing")
            
            # Clean and preprocess content
            cleaned_content = self._preprocess_content(latex_content)
            
            # Try advanced parsing first if available
            if HAS_PYLATEXENC:
                try:
                    return await self._parse_with_pylatexenc(cleaned_content)
                except Exception as e:
                    self.logger.warning(f"pylatexenc parsing failed, falling back to regex: {e}")
            
            # Fallback to regex parsing
            return await self._parse_with_regex(cleaned_content)
            
        except Exception as e:
            self.logger.error(f"LaTeX parsing failed: {e}")
            raise ParseError(f"Failed to parse LaTeX content: {e}")
    
    def _preprocess_content(self, content: str) -> str:
        """Clean and preprocess LaTeX content."""
        # Remove comments (but preserve line structure)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove comments but keep the line
            comment_pos = line.find('%')
            if comment_pos >= 0:
                # Check if % is escaped
                if comment_pos == 0 or line[comment_pos - 1] != '\\':
                    line = line[:comment_pos]
            cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Normalize whitespace in math environments
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        return cleaned_content
    
    async def _parse_with_pylatexenc(self, content: str) -> ParsedContent:
        """Parse using pylatexenc for better accuracy."""
        self.logger.debug("Using pylatexenc for parsing")
        
        parsed_content = ParsedContent()
        
        try:
            # Parse the LaTeX document
            walker = LatexWalker(content)
            nodes, _, _ = walker.get_latex_matches()
            
            # Extract mathematical environments
            await self._extract_from_nodes(nodes, parsed_content)
            
        except Exception as e:
            raise ParseError(f"pylatexenc parsing failed: {e}")
        
        return parsed_content
    
    async def _parse_with_regex(self, content: str) -> ParsedContent:
        """Parse using regex patterns as fallback."""
        self.logger.debug("Using regex-based parsing")
        
        parsed_content = ParsedContent()
        
        # Find all mathematical environments
        for match in self.env_pattern.finditer(content):
            env_type = match.group(1).lower()
            optional_arg = match.group(2) or ''
            title_arg = match.group(3) or ''
            env_content = match.group(4).strip()
            
            # Create mathematical statement
            statement = MathematicalStatement(
                type=env_type,
                name=title_arg if title_arg else None,
                statement=env_content,
                line_number=content[:match.start()].count('\n') + 1
            )
            
            # Extract label if present
            label_match = self.label_pattern.search(env_content)
            if label_match:
                statement.label = label_match.group(1)
            
            # Look for associated proof
            proof_start = match.end()
            proof_match = self.proof_pattern.search(content, proof_start)
            if proof_match and proof_match.start() - proof_start < 100:  # Proof follows closely
                statement.proof = proof_match.group(2).strip()
            
            # Categorize statement
            self._categorize_statement(statement, parsed_content)
        
        # Extract standalone math expressions
        self._extract_raw_math(content, parsed_content)
        
        return parsed_content
    
    async def _extract_from_nodes(self, nodes: List, parsed_content: ParsedContent) -> None:
        """Extract mathematical content from pylatexenc nodes."""
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode):
                env_name = node.envname.lower()
                
                if env_name in self.all_environments:
                    # Extract statement content
                    content_text = node.nodelist.latex_verbatim() if node.nodelist else ""
                    
                    statement = MathematicalStatement(
                        type=env_name,
                        statement=content_text.strip()
                    )
                    
                    # Extract optional arguments (theorem name, etc.)
                    if node.optargs and len(node.optargs) > 0:
                        statement.name = node.optargs[0].latex_verbatim()
                    
                    self._categorize_statement(statement, parsed_content)
            
            # Recursively process child nodes
            if hasattr(node, 'nodelist') and node.nodelist:
                await self._extract_from_nodes(node.nodelist, parsed_content)
    
    def _categorize_statement(self, statement: MathematicalStatement, parsed_content: ParsedContent) -> None:
        """Categorize a mathematical statement into the appropriate list."""
        stmt_type = statement.type.lower()
        
        if stmt_type in {'theorem'}:
            parsed_content.theorems.append(statement)
        elif stmt_type in {'definition'}:
            parsed_content.definitions.append(statement)
        elif stmt_type in {'lemma'}:
            parsed_content.lemmas.append(statement)
        elif stmt_type in {'corollary'}:
            parsed_content.corollaries.append(statement)
        elif stmt_type in {'proposition', 'claim', 'fact'}:
            parsed_content.propositions.append(statement)
        elif stmt_type in {'example', 'remark', 'note'}:
            parsed_content.examples.append(statement)
        else:
            # Default to theorems for unknown types
            parsed_content.theorems.append(statement)
    
    def _extract_raw_math(self, content: str, parsed_content: ParsedContent) -> None:
        """Extract raw mathematical expressions."""
        # Extract display math
        for match in self.display_math_pattern.finditer(content):
            math_content = match.group(1) or match.group(2)
            if math_content and math_content.strip():
                parsed_content.raw_math.append(math_content.strip())
        
        # Extract inline math (limit to avoid too much noise)
        inline_count = 0
        for match in self.inline_math_pattern.finditer(content):
            if inline_count >= 50:  # Limit inline math extraction
                break
            math_content = match.group(1)
            if math_content and len(math_content.strip()) > 3:  # Skip trivial expressions
                parsed_content.raw_math.append(math_content.strip())
                inline_count += 1
    
    async def parse_file(self, file_path: Path) -> ParsedContent:
        """Parse a LaTeX file.
        
        Args:
            file_path: Path to the LaTeX file
            
        Returns:
            ParsedContent object with extracted mathematical content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = await self.parse(content)
            result.metadata['source_file'] = str(file_path)
            result.metadata['file_size'] = len(content)
            
            return result
            
        except Exception as e:
            raise ParseError(f"Failed to parse file {file_path}: {e}")
    
    def extract_dependencies(self, statement: MathematicalStatement) -> List[str]:
        """Extract dependencies from a mathematical statement.
        
        This method identifies references to other theorems, definitions,
        or mathematical concepts within the statement.
        """
        dependencies = []
        
        # Look for explicit references
        ref_patterns = [
            r'\\ref\{([^}]+)\}',
            r'\\eqref\{([^}]+)\}',
            r'\\cref\{([^}]+)\}',
            r'by\s+(Theorem|Lemma|Definition|Proposition)\s+([\d\.]+)',
            r'from\s+(Theorem|Lemma|Definition|Proposition)\s+([\d\.]+)',
        ]
        
        text = statement.statement + (statement.proof or '')
        
        for pattern in ref_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    dependencies.append(match.group(1))
                elif len(match.groups()) >= 2:
                    dependencies.append(f"{match.group(1)} {match.group(2)}")
        
        return list(set(dependencies))  # Remove duplicates
    
    def get_parsing_statistics(self, parsed_content: ParsedContent) -> Dict[str, Any]:
        """Get statistics about the parsed content."""
        stats = parsed_content.count_statements()
        
        # Add more detailed statistics
        all_statements = parsed_content.get_all_statements()
        
        stats.update({
            'statements_with_proofs': sum(1 for s in all_statements if s.proof),
            'statements_with_names': sum(1 for s in all_statements if s.name),
            'statements_with_labels': sum(1 for s in all_statements if s.label),
            'raw_math_expressions': len(parsed_content.raw_math),
            'average_statement_length': (
                sum(len(s.statement) for s in all_statements) / len(all_statements)
                if all_statements else 0
            ),
        })
        
        return stats
