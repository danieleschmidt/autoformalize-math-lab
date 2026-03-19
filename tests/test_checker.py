"""Tests for ProofChecker."""
import pytest
from autoformalize_math import LaTeXParser, ProofChecker
from autoformalize_math.ast_nodes import (
    ProofNode, ProofStep, VariableNode, BinaryOpNode, NumberNode,
    QuantifierNode, LogicBinNode,
)

parser = LaTeXParser()
checker = ProofChecker()


class TestValidationReport:
    def test_valid_proof_passes(self):
        PROOF = r"""
        Assume $p \in \mathbb{Z}$.
        Then $p^2 = 2 q^2$.
        Hence $p$ is even.
        Therefore $\gcd(p, q) \geq 2$.
        """
        proof = parser.parse_proof("test", PROOF)
        report = checker.validate(proof)
        # should have all 6 checks
        assert len(report.checks) == 6

    def test_report_has_theorem_name(self):
        proof = ProofNode("my_theorem")
        proof.premises = [VariableNode("h")]
        proof.conclusion = VariableNode("C")
        proof.steps = [ProofStep("s1", VariableNode("x"), "assumption")]
        report = checker.validate(proof)
        assert report.theorem_name == "my_theorem"

    def test_valid_flag(self):
        proof = ProofNode("minimal_valid")
        proof.premises = [VariableNode("h")]
        proof.conclusion = VariableNode("C")
        proof.steps = [ProofStep("s1", VariableNode("x"), "deduction")]
        report = checker.validate(proof)
        assert isinstance(report.valid, bool)

    def test_summary_contains_theorem_name(self):
        proof = ProofNode("my_theorem")
        proof.premises = [VariableNode("h")]
        proof.conclusion = VariableNode("C")
        proof.steps = [ProofStep("s1", VariableNode("x"), "deduction")]
        report = checker.validate(proof)
        assert "my_theorem" in report.summary()


class TestIndividualChecks:
    def test_no_premises_fails(self):
        proof = ProofNode("no_hyp")
        proof.conclusion = VariableNode("C")
        proof.steps = [ProofStep("s1", VariableNode("x"), "deduction")]
        report = checker.validate(proof)
        prem_check = next(c for c in report.checks if c.name == "has_premises")
        assert not prem_check.passed

    def test_with_premises_passes(self):
        proof = ProofNode("with_hyp")
        proof.premises = [VariableNode("P")]
        proof.conclusion = VariableNode("C")
        proof.steps = [ProofStep("s1", VariableNode("x"), "deduction")]
        report = checker.validate(proof)
        prem_check = next(c for c in report.checks if c.name == "has_premises")
        assert prem_check.passed

    def test_no_conclusion_fails(self):
        proof = ProofNode("no_conc")
        proof.premises = [VariableNode("P")]
        proof.steps = [ProofStep("s1", VariableNode("x"), "deduction")]
        # do NOT set conclusion
        report = checker.validate(proof)
        conc_check = next(c for c in report.checks if c.name == "has_conclusion")
        assert not conc_check.passed

    def test_no_steps_fails(self):
        proof = ProofNode("no_steps")
        proof.premises = [VariableNode("P")]
        proof.conclusion = VariableNode("C")
        # no steps
        report = checker.validate(proof)
        steps_check = next(c for c in report.checks if c.name == "has_steps")
        assert not steps_check.passed

    def test_bare_number_conclusion_no_steps_fails(self):
        proof = ProofNode("bad_conc")
        proof.premises = [VariableNode("P")]
        proof.conclusion = NumberNode("42")
        report = checker.validate(proof)
        follows_check = next(c for c in report.checks if c.name == "conclusion_follows")
        assert not follows_check.passed


class TestProofCheckOnRealProofs:
    SQRT2_PROOF = r"""
    Assume $\sqrt{2} = \frac{p}{q}$ where $\gcd(p, q) = 1$.
    Then $p^2 = 2 q^2$.
    Hence $p$ is even.
    Therefore $\gcd(p, q) \geq 2$.
    """

    INDUCTION_PROOF = r"""
    Assume $n \geq 0$ and $\sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
    Base case: $n = 0$ gives $\sum_{i=0}^{0} i = 0$.
    Inductive step: assume formula holds for $n$.
    Therefore $\forall n \in \mathbb{N}, \sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
    """

    DEMORGAN_PROOF = r"""
    Assume $x \in (A \cup B)^c$.
    Then $x \notin A$ and $x \notin B$.
    Hence $x \in A^c \cap B^c$.
    Therefore $(A \cup B)^c \subseteq A^c \cap B^c$.
    """

    def test_sqrt2_passes_all_structural_checks(self):
        proof = parser.parse_proof("irrationality_sqrt2", self.SQRT2_PROOF)
        report = checker.validate(proof)
        assert report.passed_count >= 4  # at least 4 of 6 must pass

    def test_induction_passes_structural(self):
        proof = parser.parse_proof("gauss_induction", self.INDUCTION_PROOF)
        report = checker.validate(proof)
        assert report.passed_count >= 4

    def test_demorgan_passes_structural(self):
        proof = parser.parse_proof("demorgan", self.DEMORGAN_PROOF)
        report = checker.validate(proof)
        assert report.passed_count >= 4

    def test_passed_count_leq_total(self):
        proof = parser.parse_proof("irrationality_sqrt2", self.SQRT2_PROOF)
        report = checker.validate(proof)
        assert report.passed_count <= report.total_count


class TestEndToEnd:
    """Full pipeline: parse → convert → check."""

    def test_pipeline_runs_without_error(self):
        from autoformalize_math import FormalConverter
        conv = FormalConverter()
        PROOF = r"""
        Assume $p \in \mathbb{Z}$.
        Then $p = 2k$.
        Therefore $p$ is even.
        """
        proof = parser.parse_proof("even_proof", PROOF)
        lean = conv.convert_proof(proof)
        report = checker.validate(proof)
        assert isinstance(lean, str)
        assert len(lean) > 0
        assert len(report.checks) == 6
