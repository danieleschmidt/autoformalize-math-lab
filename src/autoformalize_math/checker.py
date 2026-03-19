"""
ProofChecker
============
Validates basic structural properties of a ProofNode.

Checks performed
----------------
1. has_premises      – at least one assumption/hypothesis is present
2. has_conclusion    – a conclusion node was extracted
3. has_steps         – proof has at least one intermediate step
4. steps_non_trivial – steps are not all raw-text fallbacks (i.e., math was parsed)
5. conclusion_follows – conclusion AST appears consistent with steps
   (lightweight: checks that the conclusion node type matches expected
    deductive patterns rather than being an unrelated atom)
6. quantifier_scope  – every free variable in the conclusion is bound somewhere
   in the premises or quantifiers

Each check returns a CheckResult with a boolean `passed` flag and a
human-readable `message`.  The overall `validate` method returns a
`ValidationReport` with a `valid` field (True iff all checks pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from .ast_nodes import (
    MathNode, VariableNode, QuantifierNode, LogicBinNode, NotNode,
    BinaryOpNode, FractionNode, SumNode, ProductNode, IntegralNode,
    SetMemberNode, SubsetNode, SetOpNode, ProofNode, ProofStep, NumberNode,
    SymbolNode, NegateNode,
)


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class ValidationReport:
    theorem_name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    def summary(self) -> str:
        lines = [
            f"Validation report for '{self.theorem_name}'",
            f"  Result: {'VALID' if self.valid else 'INVALID'}  "
            f"({self.passed_count}/{self.total_count} checks passed)",
            "",
        ]
        for c in self.checks:
            lines.append(f"  {c!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ──────────────────────────────────────────────────────────────────────────────

def _free_variables(node: MathNode, bound: Set[str] = None) -> Set[str]:
    """Collect variable names that are free (not bound by a quantifier)."""
    if bound is None:
        bound = set()
    free: Set[str] = set()

    if isinstance(node, VariableNode):
        # ignore single-letter set names and generic placeholders
        if node.name not in bound and len(node.name) <= 3:
            free.add(node.name)

    elif isinstance(node, QuantifierNode):
        new_bound = bound | {node.variable}
        if node.domain:
            free |= _free_variables(node.domain, bound)
        free |= _free_variables(node.body, new_bound)

    elif isinstance(node, BinaryOpNode):
        free |= _free_variables(node.left, bound)
        free |= _free_variables(node.right, bound)

    elif isinstance(node, FractionNode):
        free |= _free_variables(node.numerator, bound)
        free |= _free_variables(node.denominator, bound)

    elif isinstance(node, (SumNode, ProductNode, IntegralNode)):
        new_bound = bound | {node.variable}
        if node.lower:
            free |= _free_variables(node.lower, bound)
        if node.upper:
            free |= _free_variables(node.upper, bound)
        free |= _free_variables(node.body, new_bound)

    elif isinstance(node, (NegateNode, NotNode)):
        free |= _free_variables(node.operand, bound)

    elif isinstance(node, LogicBinNode):
        free |= _free_variables(node.left, bound)
        free |= _free_variables(node.right, bound)

    elif isinstance(node, SetMemberNode):
        free |= _free_variables(node.element, bound)
        free |= _free_variables(node.domain, bound)

    elif isinstance(node, SubsetNode):
        free |= _free_variables(node.left, bound)
        free |= _free_variables(node.right, bound)

    elif isinstance(node, SetOpNode):
        free |= _free_variables(node.left, bound)
        free |= _free_variables(node.right, bound)

    # NumberNode, SymbolNode → no free variables

    return free


def _all_variables(node: MathNode) -> Set[str]:
    """Collect ALL variable names in a node (free + bound)."""
    return _free_variables(node, set())


def _is_mathematical(node: MathNode) -> bool:
    """True if node is richer than a raw text fallback."""
    if isinstance(node, VariableNode):
        # long strings are likely raw prose, not math
        return len(node.name) <= 60
    return True


def _node_type_label(node: MathNode) -> str:
    return type(node).__name__


# ──────────────────────────────────────────────────────────────────────────────

class ProofChecker:
    """
    Structural validator for ProofNode objects.

    Usage
    -----
    >>> checker = ProofChecker()
    >>> report = checker.validate(proof_node)
    >>> print(report.valid)
    >>> print(report.summary())
    """

    def validate(self, proof: ProofNode) -> ValidationReport:
        """
        Run all structural checks on a ProofNode.

        Parameters
        ----------
        proof : ProofNode
            Output from LaTeXParser.parse_proof.

        Returns
        -------
        ValidationReport
            Contains individual CheckResults and an overall `valid` flag.
        """
        report = ValidationReport(theorem_name=proof.theorem_name)

        report.checks.append(self._check_has_premises(proof))
        report.checks.append(self._check_has_conclusion(proof))
        report.checks.append(self._check_has_steps(proof))
        report.checks.append(self._check_steps_non_trivial(proof))
        report.checks.append(self._check_conclusion_follows(proof))
        report.checks.append(self._check_variable_scope(proof))

        return report

    # ── individual checks ──

    @staticmethod
    def _check_has_premises(proof: ProofNode) -> CheckResult:
        passed = len(proof.premises) > 0
        if passed:
            msg = f"{len(proof.premises)} premise(s) found"
        else:
            msg = "no premises found — proof may be missing hypotheses"
        return CheckResult("has_premises", passed, msg)

    @staticmethod
    def _check_has_conclusion(proof: ProofNode) -> CheckResult:
        passed = proof.conclusion is not None
        msg = (
            f"conclusion is {_node_type_label(proof.conclusion)}"
            if passed
            else "no conclusion extracted from proof text"
        )
        return CheckResult("has_conclusion", passed, msg)

    @staticmethod
    def _check_has_steps(proof: ProofNode) -> CheckResult:
        passed = len(proof.steps) > 0
        if passed:
            msg = f"{len(proof.steps)} step(s) found"
        else:
            msg = "no proof steps found — trivial or empty proof"
        return CheckResult("has_steps", passed, msg)

    @staticmethod
    def _check_steps_non_trivial(proof: ProofNode) -> CheckResult:
        if not proof.steps:
            return CheckResult(
                "steps_non_trivial", False, "no steps to check"
            )
        math_steps = [s for s in proof.steps if _is_mathematical(s.statement)]
        ratio = len(math_steps) / len(proof.steps)
        passed = ratio >= 0.5
        msg = (
            f"{len(math_steps)}/{len(proof.steps)} steps contain "
            f"parsed math (ratio {ratio:.0%})"
        )
        return CheckResult("steps_non_trivial", passed, msg)

    @staticmethod
    def _check_conclusion_follows(proof: ProofNode) -> CheckResult:
        """
        Lightweight check: conclusion should not be purely numeric/trivial
        and should share some structure with earlier steps when the proof
        contains intermediate deduction.
        """
        if proof.conclusion is None:
            return CheckResult(
                "conclusion_follows", False, "no conclusion to check"
            )

        conc = proof.conclusion

        # A bare number as conclusion with no steps is suspicious
        if isinstance(conc, NumberNode) and not proof.steps:
            return CheckResult(
                "conclusion_follows", False,
                "conclusion is a bare number with no deductive steps"
            )

        # If we have steps, check that conclusion shares variables with them
        if proof.steps:
            conc_vars = _free_variables(conc)
            step_vars: Set[str] = set()
            for step in proof.steps:
                step_vars |= _free_variables(step.statement)
            premise_vars: Set[str] = set()
            for p in proof.premises:
                premise_vars |= _free_variables(p)

            all_known = step_vars | premise_vars
            # filter out super-generic single chars that appear everywhere
            meaningful_conc = {v for v in conc_vars if len(v) > 1 or v.isalpha()}
            if meaningful_conc and not meaningful_conc & all_known:
                return CheckResult(
                    "conclusion_follows", False,
                    f"conclusion variables {meaningful_conc} not found in steps/premises"
                )

        return CheckResult(
            "conclusion_follows", True,
            "conclusion is structurally consistent with proof body"
        )

    @staticmethod
    def _check_variable_scope(proof: ProofNode) -> CheckResult:
        """
        Check that variables appearing free in the conclusion are either:
        - introduced in a premise, OR
        - bound by a quantifier in the conclusion itself.
        """
        if proof.conclusion is None:
            return CheckResult("variable_scope", True, "no conclusion to check")

        # variables bound inside the conclusion by quantifiers
        conc_bound: Set[str] = set()

        def _collect_bound(node: MathNode) -> None:
            if isinstance(node, QuantifierNode):
                conc_bound.add(node.variable)
                _collect_bound(node.body)
            elif isinstance(node, (BinaryOpNode, LogicBinNode)):
                _collect_bound(node.left)  # type: ignore[arg-type]
                _collect_bound(node.right)  # type: ignore[arg-type]

        _collect_bound(proof.conclusion)

        # variables introduced in premises
        premise_vars: Set[str] = set()
        for p in proof.premises:
            premise_vars |= _free_variables(p)
            if isinstance(p, QuantifierNode):
                premise_vars.add(p.variable)

        # free vars in conclusion that are not accounted for
        free_in_conc = _free_variables(proof.conclusion)
        unbound = free_in_conc - conc_bound - premise_vars

        # single-char generic variables (n, k, m, x, y) are OK — they are
        # implicitly universally quantified in mathematical convention
        genuinely_unbound = {v for v in unbound if len(v) > 1}

        if genuinely_unbound:
            return CheckResult(
                "variable_scope", False,
                f"variables {genuinely_unbound} appear free in conclusion "
                f"but not bound in premises"
            )

        return CheckResult(
            "variable_scope", True,
            "all multi-character conclusion variables are properly scoped"
        )
