"""Unit tests for mathematical content parsers."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from autoformalize.parsers.latex import LaTeXParser
from autoformalize.parsers.pdf import PDFParser
from autoformalize.parsers.arxiv import ArXivParser
from tests.fixtures import ALL_THEOREMS, get_sample_latex_errors


class TestLaTeXParser:
    """Test cases for LaTeX parser functionality."""
    
    def test_parse_simple_theorem(self, sample_latex_theorem):
        """Test parsing a simple LaTeX theorem."""
        parser = LaTeXParser()
        result = parser.parse(sample_latex_theorem)
        
        assert result is not None
        assert "theorems" in result
        assert len(result["theorems"]) >= 1
        assert "Fundamental Theorem of Arithmetic" in str(result)
    
    def test_parse_theorem_with_proof(self):
        """Test parsing theorem with proof."""
        theorem_data = ALL_THEOREMS["quadratic_formula"]
        parser = LaTeXParser()
        result = parser.parse(theorem_data["latex"])
        
        assert result is not None
        assert "theorems" in result
        assert "proofs" in result
        assert len(result["proofs"]) >= 1
    
    def test_parse_multiple_theorems(self):
        """Test parsing multiple theorems in one document.""" 
        combined_latex = ""
        for theorem in ["quadratic_formula", "binomial_theorem"]:
            combined_latex += ALL_THEOREMS[theorem]["latex"] + "\n\n"
        
        parser = LaTeXParser()
        result = parser.parse(combined_latex)
        
        assert result is not None
        assert len(result["theorems"]) >= 2
    
    def test_parse_empty_input(self):
        """Test parsing empty or whitespace-only input."""
        parser = LaTeXParser()
        
        # Test empty string
        result = parser.parse("")
        assert result["theorems"] == []
        
        # Test whitespace only
        result = parser.parse("   \n\t  ")
        assert result["theorems"] == []
    
    def test_parse_invalid_latex(self):
        """Test handling of invalid LaTeX syntax."""
        parser = LaTeXParser()
        errors = get_sample_latex_errors()
        
        for error_type, invalid_latex in errors.items():
            with pytest.raises(Exception) or pytest.warns(UserWarning):
                parser.parse(invalid_latex)
    
    def test_extract_mathematical_expressions(self):
        """Test extraction of mathematical expressions."""
        latex_with_math = r"""
        Let $x = \frac{a}{b}$ and $y = \sqrt{c}$.
        Then we have: $$\int_0^1 f(x) dx = \sum_{n=1}^{\infty} \frac{1}{n^2}$$
        """
        
        parser = LaTeXParser()
        result = parser.parse(latex_with_math)
        
        assert "math_expressions" in result
        assert len(result["math_expressions"]) >= 2
    
    def test_extract_definitions(self):
        """Test extraction of mathematical definitions."""
        definition_latex = r"""
        \begin{definition}[Group]
        A group is a set $G$ together with a binary operation $*$ such that:
        \begin{enumerate}
        \item Closure: For all $a, b \in G$, $a * b \in G$
        \item Associativity: For all $a, b, c \in G$, $(a * b) * c = a * (b * c)$
        \item Identity: There exists $e \in G$ such that $e * a = a * e = a$ for all $a \in G$
        \item Inverse: For all $a \in G$, there exists $a^{-1} \in G$ such that $a * a^{-1} = a^{-1} * a = e$
        \end{enumerate}
        \end{definition}
        """
        
        parser = LaTeXParser()
        result = parser.parse(definition_latex)
        
        assert "definitions" in result
        assert len(result["definitions"]) >= 1
        assert "Group" in str(result["definitions"][0])


class TestPDFParser:
    """Test cases for PDF parser functionality."""
    
    @patch('autoformalize.parsers.pdf.fitz.open')
    def test_parse_pdf_file(self, mock_fitz_open):
        """Test parsing a PDF file."""
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample theorem content"
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        parser = PDFParser()
        result = parser.parse_file(Path("dummy.pdf"))
        
        assert result is not None
        assert "content" in result
        assert "Sample theorem content" in result["content"]
    
    def test_parse_nonexistent_file(self):
        """Test handling of nonexistent PDF files."""
        parser = PDFParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("nonexistent.pdf"))
    
    @patch('autoformalize.parsers.pdf.fitz.open')
    def test_parse_corrupted_pdf(self, mock_fitz_open):
        """Test handling of corrupted PDF files."""
        mock_fitz_open.side_effect = Exception("Corrupted PDF")
        
        parser = PDFParser()
        
        with pytest.raises(Exception):
            parser.parse_file(Path("corrupted.pdf"))


class TestArXivParser:
    """Test cases for arXiv parser functionality."""
    
    @patch('requests.get')
    def test_fetch_arxiv_paper(self, mock_get):
        """Test fetching paper from arXiv."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"Sample arXiv paper content"
        mock_get.return_value = mock_response
        
        parser = ArXivParser()
        result = parser.fetch_paper("2301.00001")
        
        assert result is not None
        assert len(result) > 0
    
    @patch('requests.get')
    def test_fetch_invalid_arxiv_id(self, mock_get):
        """Test handling of invalid arXiv IDs."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        parser = ArXivParser()
        
        with pytest.raises(Exception):
            parser.fetch_paper("invalid.id")
    
    def test_parse_arxiv_id_formats(self):
        """Test parsing different arXiv ID formats."""
        parser = ArXivParser()
        
        # Test various valid formats
        valid_ids = [
            "2301.00001",
            "2301.00001v1",
            "math.NT/0123456",
            "math-ph/0123456v2"
        ]
        
        for arxiv_id in valid_ids:
            normalized_id = parser.normalize_arxiv_id(arxiv_id)
            assert normalized_id is not None
            assert len(normalized_id) > 0
    
    def test_extract_metadata(self):
        """Test extraction of paper metadata from arXiv."""
        sample_xml = """
        <entry>
            <title>Sample Mathematical Paper</title>
            <author><name>John Doe</name></author>
            <published>2023-01-01T00:00:00Z</published>
            <summary>This paper proves important theorems.</summary>
        </entry>
        """
        
        parser = ArXivParser()
        metadata = parser.extract_metadata(sample_xml)
        
        assert metadata["title"] == "Sample Mathematical Paper"
        assert metadata["authors"] == ["John Doe"]
        assert "2023" in metadata["published"]


@pytest.mark.mathematical
class TestMathematicalAccuracy:
    """Test mathematical accuracy of parsing."""
    
    def test_preserve_mathematical_notation(self):
        """Test that mathematical notation is preserved correctly."""
        complex_math = r"""
        \begin{theorem}
        For the integral $\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$,
        we use the substitution $u = x^2$.
        \end{theorem}
        """
        
        parser = LaTeXParser()
        result = parser.parse(complex_math)
        
        # Check that key mathematical symbols are preserved
        content = str(result)
        assert "\\int" in content
        assert "\\infty" in content
        assert "\\sqrt{\\pi}" in content
    
    def test_parse_theorem_numbering(self):
        """Test parsing of theorem numbering and references."""
        numbered_theorems = r"""
        \begin{theorem}\label{thm:main}
        This is the main theorem.
        \end{theorem}
        
        \begin{lemma}\label{lem:helper}
        This lemma supports Theorem~\ref{thm:main}.  
        \end{lemma}
        """
        
        parser = LaTeXParser()
        result = parser.parse(numbered_theorems)
        
        assert len(result["theorems"]) >= 1
        assert len(result["lemmas"]) >= 1
        # Check that references are captured
        assert "references" in result