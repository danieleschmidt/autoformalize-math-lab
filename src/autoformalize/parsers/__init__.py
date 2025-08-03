"""LaTeX and mathematical content parsing.

This module provides parsers for extracting mathematical content from various
formats including LaTeX, PDF, and structured mathematical documents.

Modules:
    latex_parser: LaTeX document and mathematical expression parsing
    pdf_parser: PDF document content extraction (planned)
    arxiv_parser: ArXiv paper processing (planned)
    math_extractor: Mathematical theorem and proof extraction (planned)
    semantic_analyzer: Mathematical concept and dependency analysis (planned)
"""

# Import implemented parsers
try:
    from .latex_parser import LaTeXParser, ParsedContent, MathematicalStatement
    HAS_LATEX_PARSER = True
except ImportError:
    HAS_LATEX_PARSER = False

__all__ = [
    "LaTeXParser",
    "ParsedContent", 
    "MathematicalStatement",
]

# Legacy compatibility
if HAS_LATEX_PARSER:
    # Alias for backwards compatibility
    LatexParser = LaTeXParser