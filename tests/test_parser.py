"""Tests for LaTeXParser."""
import pytest
from autoformalize_math import LaTeXParser
from autoformalize_math.ast_nodes import (
    FractionNode, SumNode, ProductNode, IntegralNode, BinaryOpNode,
    QuantifierNode, LogicBinNode, NotNode, SetMemberNode, SubsetNode,
    SetOpNode, NumberNode, VariableNode, NegateNode, ProofNode,
)

parser = LaTeXParser()


class TestFraction:
    def test_basic_frac(self):
        node = parser.parse_expression(r"\frac{1}{2}")
        assert isinstance(node, FractionNode)
        assert isinstance(node.numerator, NumberNode)
        assert node.numerator.value == "1"
        assert isinstance(node.denominator, NumberNode)
        assert node.denominator.value == "2"

    def test_nested_frac(self):
        node = parser.parse_expression(r"\frac{\frac{a}{b}}{c}")
        assert isinstance(node, FractionNode)
        assert isinstance(node.numerator, FractionNode)

    def test_frac_with_vars(self):
        node = parser.parse_expression(r"\frac{p}{q}")
        assert isinstance(node, FractionNode)
        assert isinstance(node.numerator, VariableNode)
        assert node.numerator.name == "p"


class TestSumProductIntegral:
    def test_sum(self):
        node = parser.parse_expression(r"\sum_{i=0}^{n} i")
        assert isinstance(node, SumNode)
        assert node.variable == "i"

    def test_prod(self):
        node = parser.parse_expression(r"\prod_{k=1}^{n} k")
        assert isinstance(node, ProductNode)
        assert node.variable == "k"

    def test_integral(self):
        node = parser.parse_expression(r"\int_{0}^{1} x")
        assert isinstance(node, IntegralNode)

    def test_sum_body_is_var(self):
        # \sum_{i=0}^{n} i^2 — the braced form ensures the body is i^2
        node = parser.parse_expression(r"\sum_{i=0}^{n} {i^2}")
        assert isinstance(node, SumNode)
        # body should be a power expression
        assert isinstance(node.body, BinaryOpNode)
        assert node.body.op == "^"


class TestLogic:
    def test_forall(self):
        node = parser.parse_expression(r"\forall x, P")
        assert isinstance(node, QuantifierNode)
        assert node.kind == "forall"
        assert node.variable == "x"

    def test_exists(self):
        node = parser.parse_expression(r"\exists x, Q")
        assert isinstance(node, QuantifierNode)
        assert node.kind == "exists"

    def test_implies(self):
        node = parser.parse_expression(r"P \implies Q")
        assert isinstance(node, LogicBinNode)
        assert node.op == "→"

    def test_iff(self):
        node = parser.parse_expression(r"P \iff Q")
        assert isinstance(node, LogicBinNode)
        assert node.op == "↔"

    def test_land(self):
        node = parser.parse_expression(r"A \land B")
        assert isinstance(node, LogicBinNode)
        assert node.op == "∧"

    def test_lor(self):
        node = parser.parse_expression(r"A \lor B")
        assert isinstance(node, LogicBinNode)
        assert node.op == "∨"

    def test_neg(self):
        node = parser.parse_expression(r"\neg P")
        assert isinstance(node, NotNode)

    def test_forall_with_domain(self):
        node = parser.parse_expression(r"\forall n \in \mathbb{N}, P")
        assert isinstance(node, QuantifierNode)
        assert node.domain is not None


class TestSets:
    def test_in(self):
        node = parser.parse_expression(r"x \in S")
        assert isinstance(node, SetMemberNode)

    def test_subseteq(self):
        node = parser.parse_expression(r"A \subseteq B")
        assert isinstance(node, SubsetNode)

    def test_cup(self):
        node = parser.parse_expression(r"A \cup B")
        assert isinstance(node, SetOpNode)
        assert node.op == "∪"

    def test_cap(self):
        node = parser.parse_expression(r"A \cap B")
        assert isinstance(node, SetOpNode)
        assert node.op == "∩"


class TestArithmetic:
    def test_addition(self):
        node = parser.parse_expression("a + b")
        assert isinstance(node, BinaryOpNode)
        assert node.op == "+"

    def test_power(self):
        node = parser.parse_expression("x^2")
        assert isinstance(node, BinaryOpNode)
        assert node.op == "^"
        assert isinstance(node.left, VariableNode)
        assert isinstance(node.right, NumberNode)

    def test_negation(self):
        node = parser.parse_expression("-x")
        assert isinstance(node, NegateNode)

    def test_equality(self):
        node = parser.parse_expression("x = 0")
        assert isinstance(node, BinaryOpNode)
        assert node.op == "="


class TestProofParsing:
    SQRT2_PROOF = r"""
    Assume $\sqrt{2} = \frac{p}{q}$ where $\gcd(p, q) = 1$.
    Then $p^2 = 2 q^2$.
    Hence $p$ is even.
    Therefore $\gcd(p, q) \geq 2$, contradicting $\gcd(p, q) = 1$.
    """

    def test_returns_proof_node(self):
        proof = parser.parse_proof("test_theorem", self.SQRT2_PROOF)
        assert isinstance(proof, ProofNode)
        assert proof.theorem_name == "test_theorem"

    def test_has_premises(self):
        proof = parser.parse_proof("test_theorem", self.SQRT2_PROOF)
        assert len(proof.premises) >= 1

    def test_has_conclusion(self):
        proof = parser.parse_proof("test_theorem", self.SQRT2_PROOF)
        assert proof.conclusion is not None

    def test_has_steps(self):
        proof = parser.parse_proof("test_theorem", self.SQRT2_PROOF)
        assert len(proof.steps) >= 1

    def test_dollar_stripped(self):
        """Parse an expression wrapped in $ signs."""
        node = parser.parse_expression("$x + y$")
        assert isinstance(node, BinaryOpNode)
        assert node.op == "+"
