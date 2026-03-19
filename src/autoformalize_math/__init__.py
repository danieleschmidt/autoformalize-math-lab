"""
autoformalize-math-lab
======================
Rule-based LaTeX → Lean4-style proof formalization.

Components:
  - LaTeXParser: parses LaTeX math into an AST
  - FormalConverter: converts AST → Lean4-style pseudocode
  - ProofChecker: validates structural proof properties
"""

from .parser import LaTeXParser
from .converter import FormalConverter
from .checker import ProofChecker

__all__ = ["LaTeXParser", "FormalConverter", "ProofChecker"]
__version__ = "0.1.0"
