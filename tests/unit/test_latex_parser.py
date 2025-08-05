"""Unit tests for LaTeX parser."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from autoformalize.parsers.latex_parser import LaTeXParser, ParsedContent, MathematicalStatement
from autoformalize.core.exceptions import ParseError


class TestMathematicalStatement:
    """Test cases for MathematicalStatement dataclass."""

    def test_mathematical_statement_creation(self):
        """Test creating a mathematical statement."""
        statement = MathematicalStatement(
            type="theorem",
            name="test_theorem",
            statement="For all n, n + 0 = n",
            proof="By the definition of addition",
            line_number=10
        )
        
        assert statement.type == "theorem"
        assert statement.name == "test_theorem"
        assert statement.statement == "For all n, n + 0 = n"
        assert statement.proof == "By the definition of addition"
        assert statement.line_number == 10

    def test_mathematical_statement_str(self):
        """Test string representation of mathematical statement."""
        statement = MathematicalStatement(
            type="theorem",
            name="test_theorem",
            statement="This is a very long mathematical statement that should be truncated in the string representation"
        )
        
        str_repr = str(statement)
        assert "Theorem (test_theorem)" in str_repr
        assert len(str_repr) < 200  # Should be truncated


class TestParsedContent:
    """Test cases for ParsedContent dataclass."""

    def test_parsed_content_creation(self):
        """Test creating parsed content."""
        theorem = MathematicalStatement(type="theorem", statement="test theorem")
        definition = MathematicalStatement(type="definition", statement="test definition")
        
        parsed = ParsedContent(
            theorems=[theorem],
            definitions=[definition],
            raw_math=["$x + y = z$"]
        )
        
        assert len(parsed.theorems) == 1
        assert len(parsed.definitions) == 1
        assert len(parsed.raw_math) == 1

    def test_get_all_statements(self):
        """Test getting all statements from parsed content."""
        theorem = MathematicalStatement(type="theorem", statement="test theorem")
        lemma = MathematicalStatement(type="lemma", statement="test lemma")
        definition = MathematicalStatement(type="definition", statement="test definition")
        
        parsed = ParsedContent(
            theorems=[theorem],
            lemmas=[lemma],
            definitions=[definition]
        )
        
        all_statements = parsed.get_all_statements()
        assert len(all_statements) == 3
        assert theorem in all_statements
        assert lemma in all_statements
        assert definition in all_statements

    def test_count_statements(self):
        """Test counting statements by type."""
        parsed = ParsedContent(
            theorems=[MathematicalStatement(type="theorem", statement="t1"),
                     MathematicalStatement(type="theorem", statement="t2")],
            definitions=[MathematicalStatement(type="definition", statement="d1")],
            lemmas=[MathematicalStatement(type="lemma", statement="l1")]
        )
        
        counts = parsed.count_statements()
        assert counts["theorems"] == 2
        assert counts["definitions"] == 1
        assert counts["lemmas"] == 1
        assert counts["corollaries"] == 0
        assert counts["total"] == 4


class TestLaTeXParser:
    """Test cases for LaTeX parser."""

    @pytest.fixture
    def parser(self):
        """Create a LaTeX parser for testing."""
        return LaTeXParser()

    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.MATH_ENVIRONMENTS == {
            'theorem', 'lemma', 'corollary', 'proposition', 'definition',
            'example', 'remark', 'note', 'claim', 'fact', 'observation',
            'conjecture', 'hypothesis', 'axiom', 'postulate'
        }
        assert parser.PROOF_ENVIRONMENTS == {'proof', 'solution', 'sketch'}

    def test_parser_custom_environments(self):
        """Test parser with custom environments."""
        custom_envs = {'custom_theorem', 'special_lemma'}
        parser = LaTeXParser(custom_environments=custom_envs)
        
        assert 'custom_theorem' in parser.all_environments
        assert 'special_lemma' in parser.all_environments
        assert 'theorem' in parser.all_environments  # Still includes defaults

    @pytest.mark.asyncio
    async def test_parse_simple_theorem(self, parser):
        """Test parsing a simple theorem."""
        latex_content = """
        \\begin{theorem}
        For any natural number $n$, we have $n + 0 = n$.
        \\end{theorem}
        """
        
        result = await parser.parse(latex_content)
        
        assert len(result.theorems) == 1
        theorem = result.theorems[0]
        assert theorem.type == "theorem"
        assert "n + 0 = n" in theorem.statement

    @pytest.mark.asyncio
    async def test_parse_theorem_with_proof(self, parser):
        """Test parsing theorem with associated proof."""
        latex_content = """
        \\begin{theorem}
        For any natural number $n$, we have $n + 0 = n$.
        \\end{theorem}
        \\begin{proof}
        This follows from the definition of addition.
        \\end{proof}
        """
        
        result = await parser.parse(latex_content)
        
        assert len(result.theorems) == 1
        theorem = result.theorems[0]
        assert theorem.proof is not None
        assert "definition of addition" in theorem.proof

    @pytest.mark.asyncio
    async def test_parse_multiple_environments(self, parser):
        """Test parsing multiple mathematical environments."""
        latex_content = """
        \\begin{definition}
        A group is a set with an associative binary operation.
        \\end{definition}
        
        \\begin{theorem}[Fundamental Theorem]
        Every finite group has a Cayley table.
        \\end{theorem}
        
        \\begin{lemma}
        Subgroups are closed under the group operation.
        \\end{lemma}
        """
        
        result = await parser.parse(latex_content)
        
        assert len(result.definitions) == 1
        assert len(result.theorems) == 1
        assert len(result.lemmas) == 1
        
        # Check theorem name extraction
        theorem = result.theorems[0]
        assert theorem.name == "Fundamental Theorem"

    @pytest.mark.asyncio
    async def test_parse_with_labels(self, parser):
        """Test parsing environments with labels."""
        latex_content = """
        \\begin{theorem}\\label{thm:main}
        This is the main theorem.
        \\end{theorem}
        """
        
        result = await parser.parse(latex_content)
        
        assert len(result.theorems) == 1
        theorem = result.theorems[0]
        assert theorem.label == "thm:main"

    @pytest.mark.asyncio
    async def test_parse_inline_and_display_math(self, parser):
        """Test parsing inline and display math."""
        latex_content = """
        Consider the equation $x^2 + y^2 = z^2$.
        
        We also have the display equation:
        $$\\int_0^1 x^2 dx = \\frac{1}{3}$$
        
        And another display equation:
        \\[\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}\\]
        """
        
        result = await parser.parse(latex_content)
        
        assert len(result.raw_math) >= 2  # Should extract multiple math expressions
        math_expressions = ' '.join(result.raw_math)
        assert "x^2 + y^2 = z^2" in math_expressions

    @pytest.mark.asyncio
    async def test_parse_empty_content(self, parser):
        """Test parsing empty content."""
        result = await parser.parse("")
        
        assert len(result.theorems) == 0
        assert len(result.definitions) == 0
        assert len(result.raw_math) == 0

    @pytest.mark.asyncio
    async def test_parse_content_with_comments(self, parser):
        """Test parsing content with LaTeX comments."""
        latex_content = """
        % This is a comment
        \\begin{theorem}
        This is a theorem. % inline comment
        \\end{theorem}
        % Another comment
        """
        
        result = await parser.parse(latex_content)
        
        assert len(result.theorems) == 1
        # Comments should be removed from processing

    @pytest.mark.asyncio
    async def test_parse_malformed_latex_graceful_failure(self, parser):
        """Test that malformed LaTeX fails gracefully."""
        latex_content = """
        \\begin{theorem}
        This theorem is not closed properly
        \\begin{lemma}
        Neither is this lemma
        """
        
        # Should not raise exception but may have warnings
        result = await parser.parse(latex_content)
        
        # Parser should attempt to extract what it can
        assert isinstance(result, ParsedContent)

    @pytest.mark.asyncio 
    async def test_parse_file(self, parser, tmp_path):
        """Test parsing a LaTeX file."""
        # Create temporary LaTeX file
        test_file = tmp_path / "test.tex"
        test_file.write_text("""
        \\begin{theorem}
        Test theorem from file.
        \\end{theorem}
        """)
        
        result = await parser.parse_file(test_file)
        
        assert len(result.theorems) == 1
        assert result.metadata["source_file"] == str(test_file)
        assert "file_size" in result.metadata

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self, parser):
        """Test parsing non-existent file."""
        with pytest.raises(ParseError):
            await parser.parse_file("nonexistent.tex")

    def test_extract_dependencies(self, parser):
        """Test extracting dependencies from statements."""
        statement = MathematicalStatement(
            type="theorem",
            statement="By Theorem 2.1 and using \\ref{lemma:main}, we prove...",
            proof="From Lemma 3.5 and Definition 1.2..."
        )
        
        dependencies = parser.extract_dependencies(statement)
        
        assert "lemma:main" in dependencies
        # Should extract references from both statement and proof

    def test_get_parsing_statistics(self, parser):
        """Test getting parsing statistics."""
        parsed_content = ParsedContent(
            theorems=[
                MathematicalStatement(type="theorem", statement="t1", name="Theorem 1"),
                MathematicalStatement(type="theorem", statement="t2", proof="proof")
            ],
            definitions=[
                MathematicalStatement(type="definition", statement="d1", label="def:1")
            ],
            raw_math=["$x = y$", "$a + b = c$"]
        )
        
        stats = parser.get_parsing_statistics(parsed_content)
        
        assert stats["theorems"] == 2
        assert stats["definitions"] == 1
        assert stats["total"] == 3
        assert stats["statements_with_proofs"] == 1
        assert stats["statements_with_names"] == 1
        assert stats["statements_with_labels"] == 1
        assert stats["raw_math_expressions"] == 2
        assert "average_statement_length" in stats


@pytest.mark.asyncio
class TestLaTeXParserIntegration:
    """Integration tests for LaTeX parser."""

    async def test_complex_document_parsing(self):
        """Test parsing a complex LaTeX document."""
        parser = LaTeXParser()
        
        complex_latex = """
        \\documentclass{article}
        \\usepackage{amsmath, amsthm}
        
        \\begin{document}
        
        \\section{Group Theory}
        
        \\begin{definition}\\label{def:group}
        A \\emph{group} is a set $G$ together with a binary operation $*: G \\times G \\to G$
        such that:
        \\begin{enumerate}
        \\item (Associativity) For all $a, b, c \\in G$: $(a * b) * c = a * (b * c)$
        \\item (Identity) There exists $e \\in G$ such that $e * a = a * e = a$ for all $a \\in G$
        \\item (Inverse) For each $a \\in G$, there exists $a^{-1} \\in G$ such that $a * a^{-1} = a^{-1} * a = e$
        \\end{enumerate}
        \\end{definition}
        
        \\begin{theorem}[Lagrange's Theorem]\\label{thm:lagrange}
        If $G$ is a finite group and $H$ is a subgroup of $G$, then $|H|$ divides $|G|$.
        \\end{theorem}
        
        \\begin{proof}
        Consider the left cosets of $H$ in $G$. Each coset has exactly $|H|$ elements,
        and the cosets partition $G$. Therefore $|G| = k|H|$ for some integer $k$.
        \\end{proof}
        
        \\begin{corollary}
        If $p$ is prime and $G$ is a group of order $p$, then $G$ is cyclic.
        \\end{corollary}
        
        \\end{document}
        """
        
        result = await parser.parse(complex_latex)
        
        # Check that all environments were detected
        assert len(result.definitions) == 1
        assert len(result.theorems) == 1
        assert len(result.corollaries) == 1
        
        # Check content extraction
        definition = result.definitions[0]
        assert definition.label == "def:group"
        assert "binary operation" in definition.statement
        
        theorem = result.theorems[0]
        assert theorem.name == "Lagrange's Theorem"
        assert theorem.label == "thm:lagrange"
        assert theorem.proof is not None
        assert "cosets" in theorem.proof
        
        # Check statistics
        stats = parser.get_parsing_statistics(result)
        assert stats["total"] == 3
        assert stats["statements_with_labels"] == 2
        assert stats["statements_with_names"] == 1
        assert stats["statements_with_proofs"] == 1