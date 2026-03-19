"""Tests for FormalConverter."""
import pytest
from autoformalize_math import LaTeXParser, FormalConverter
from autoformalize_math.ast_nodes import ProofNode, NumberNode, VariableNode

parser = LaTeXParser()
converter = FormalConverter()


class TestExpressionConversion:
    def test_number(self):
        node = NumberNode("42")
        assert converter.convert_expression(node) == "42"

    def test_variable(self):
        node = VariableNode("x")
        assert converter.convert_expression(node) == "x"

    def test_set_symbols(self):
        node = VariableNode("N")
        assert converter.convert_expression(node) == "ℕ"

        node = VariableNode("Z")
        assert converter.convert_expression(node) == "ℤ"

    def test_fraction(self):
        node = parser.parse_expression(r"\frac{1}{2}")
        result = converter.convert_expression(node)
        assert "/" in result
        assert "ℚ" in result

    def test_sum(self):
        node = parser.parse_expression(r"\sum_{i=0}^{n} i")
        result = converter.convert_expression(node)
        assert "∑" in result
        assert "Finset.range" in result

    def test_product(self):
        node = parser.parse_expression(r"\prod_{k=1}^{n} k")
        result = converter.convert_expression(node)
        assert "∏" in result

    def test_integral(self):
        node = parser.parse_expression(r"\int_{0}^{1} x")
        result = converter.convert_expression(node)
        assert "∫" in result

    def test_forall(self):
        node = parser.parse_expression(r"\forall x, P")
        result = converter.convert_expression(node)
        assert "∀" in result
        assert "x" in result

    def test_exists(self):
        node = parser.parse_expression(r"\exists x, Q")
        result = converter.convert_expression(node)
        assert "∃" in result

    def test_implies(self):
        node = parser.parse_expression(r"P \implies Q")
        result = converter.convert_expression(node)
        assert "→" in result

    def test_iff(self):
        node = parser.parse_expression(r"P \iff Q")
        result = converter.convert_expression(node)
        assert "↔" in result

    def test_land(self):
        node = parser.parse_expression(r"A \land B")
        result = converter.convert_expression(node)
        assert "∧" in result

    def test_lor(self):
        node = parser.parse_expression(r"A \lor B")
        result = converter.convert_expression(node)
        assert "∨" in result

    def test_not(self):
        node = parser.parse_expression(r"\neg P")
        result = converter.convert_expression(node)
        assert "¬" in result

    def test_set_in(self):
        node = parser.parse_expression(r"x \in S")
        result = converter.convert_expression(node)
        assert "∈" in result

    def test_subseteq(self):
        node = parser.parse_expression(r"A \subseteq B")
        result = converter.convert_expression(node)
        assert "⊆" in result

    def test_cup(self):
        node = parser.parse_expression(r"A \cup B")
        result = converter.convert_expression(node)
        assert "∪" in result


class TestProofConversion:
    GAUSS_PROOF = r"""
    Assume $\sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
    Base case: $\sum_{i=0}^{0} i = 0 = \frac{0 \cdot 1}{2}$.
    Inductive step: assume $\sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
    Therefore $\forall n \in \mathbb{N}, \sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
    """

    def test_proof_contains_theorem(self):
        proof = parser.parse_proof("gauss_sum", self.GAUSS_PROOF)
        result = converter.convert_proof(proof)
        assert "theorem gauss_sum" in result

    def test_proof_contains_by(self):
        proof = parser.parse_proof("gauss_sum", self.GAUSS_PROOF)
        result = converter.convert_proof(proof)
        assert ":= by" in result

    def test_proof_contains_show(self):
        proof = parser.parse_proof("gauss_sum", self.GAUSS_PROOF)
        result = converter.convert_proof(proof)
        assert "show" in result

    def test_proof_contains_have(self):
        proof = parser.parse_proof("gauss_sum", self.GAUSS_PROOF)
        result = converter.convert_proof(proof)
        assert "have" in result

    def test_empty_proof_fallback(self):
        """An empty ProofNode should still produce valid Lean4 skeleton."""
        proof = ProofNode("empty_theorem")
        result = converter.convert_proof(proof)
        assert "theorem empty_theorem" in result
        assert "trivial" in result
