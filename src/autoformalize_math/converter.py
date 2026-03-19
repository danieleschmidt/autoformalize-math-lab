"""
FormalConverter
===============
Converts a MathNode AST into Lean4-style pseudocode.

Output format
-------------
The generated code follows Lean4 syntax conventions but is *pseudocode*:
it is designed to be readable and structurally correct for humans and
future LLM pipelines, not to type-check in actual Lean4.

Key conventions
---------------
- Theorems are declared with `theorem <name> : <type> := by`
- Quantifiers become `∀ (x : α), P x` / `∃ x, P x`
- Fractions become `(a / b : ℚ)`  (rational number coercion)
- Sums/Products become `∑ i in Finset.range n, f i` etc.
- Integrals become `∫ x in a..b, f x`
- Logic operators: `∧  ∨  →  ↔  ¬`
- Proof steps are rendered as `have <label> : <type> := by ...`
- The conclusion is the final `show` target
"""
from __future__ import annotations

from typing import Optional
from .ast_nodes import (
    MathNode, NumberNode, VariableNode, SymbolNode,
    BinaryOpNode, FractionNode, SumNode, ProductNode, IntegralNode, NegateNode,
    LogicBinNode, NotNode, QuantifierNode, SetMemberNode, SubsetNode, SetOpNode,
    ProofStep, ProofNode,
)

# ──────────────────────────────────────────────────────────────────────────────
# Set/type name mapping
# ──────────────────────────────────────────────────────────────────────────────

_SET_MAP = {
    "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "R": "ℝ", "C": "ℂ",
    "\\mathbb{N}": "ℕ", "\\mathbb{Z}": "ℤ",
    "\\mathbb{Q}": "ℚ", "\\mathbb{R}": "ℝ",
    "𝕄N": "ℕ", "𝕄Z": "ℤ", "𝕄Q": "ℚ", "𝕄R": "ℝ", "𝕄C": "ℂ",
}

_OP_MAP = {
    "+": "+", "-": "-", "*": "*", "/": "/", "^": "^",
    "=": "=", "≠": "≠", "≤": "≤", "≥": "≥", "<": "<", ">": ">",
    "\\neq": "≠", "\\leq": "≤", "\\geq": "≥",
}


class FormalConverter:
    """
    Convert a parsed MathNode AST into Lean4-style pseudocode string.

    Usage
    -----
    >>> converter = FormalConverter()
    >>> lean_expr = converter.convert_expression(ast_node)
    >>> lean_proof = converter.convert_proof(proof_node)
    """

    def convert_expression(self, node: MathNode) -> str:
        """
        Convert an expression AST node into a Lean4 expression string.

        Parameters
        ----------
        node : MathNode
            Root of the expression AST (from LaTeXParser.parse_expression).

        Returns
        -------
        str
            Lean4-style expression.
        """
        return self._expr(node)

    def convert_proof(self, proof: ProofNode) -> str:
        """
        Convert a ProofNode into a complete Lean4-style theorem + proof block.

        Parameters
        ----------
        proof : ProofNode
            Structured proof AST (from LaTeXParser.parse_proof).

        Returns
        -------
        str
            Multi-line Lean4 pseudocode.
        """
        lines: list[str] = []

        # ── hypothesis types ──
        hyp_strs = [self._expr(p) for p in proof.premises]
        hyp_decls = " ".join(
            f"(h{i+1} : {h})" for i, h in enumerate(hyp_strs)
        )

        # ── conclusion type ──
        conc_str = (
            self._expr(proof.conclusion) if proof.conclusion
            else "True"
        )

        # ── theorem signature ──
        name = proof.theorem_name.replace(" ", "_").replace("-", "_")
        if hyp_decls:
            lines.append(f"theorem {name} {hyp_decls} : {conc_str} := by")
        else:
            lines.append(f"theorem {name} : {conc_str} := by")

        # ── proof steps ──
        if not proof.steps:
            lines.append("  trivial")
        else:
            for step in proof.steps:
                step_str = self._expr(step.statement)
                just = f"  -- {step.justification}" if step.justification else ""
                lines.append(f"  have {step.label} : {step_str} := by{just}")
                lines.append(f"    exact?")

        # ── conclusion ──
        lines.append(f"  show {conc_str}")
        lines.append("  exact?")

        return "\n".join(lines)

    # ── private expression converter ──

    def _expr(self, node: MathNode) -> str:  # noqa: C901
        """Recursively convert a node to Lean4 expression string."""

        if isinstance(node, NumberNode):
            return node.value

        if isinstance(node, VariableNode):
            # pass through set names
            return _SET_MAP.get(node.name, node.name)

        if isinstance(node, SymbolNode):
            return _SET_MAP.get(node.name, node.name)

        if isinstance(node, FractionNode):
            n = self._expr(node.numerator)
            d = self._expr(node.denominator)
            return f"({n} / {d} : ℚ)"

        if isinstance(node, BinaryOpNode):
            op = _OP_MAP.get(node.op, node.op)
            left = self._paren_if_needed(node.left, node.op)
            right = self._paren_if_needed(node.right, node.op)
            return f"{left} {op} {right}"

        if isinstance(node, NegateNode):
            return f"-{self._expr(node.operand)}"

        if isinstance(node, NotNode):
            return f"¬{self._expr(node.operand)}"

        if isinstance(node, SumNode):
            body = self._expr(node.body)
            lower = self._bound(node.lower)
            upper = self._bound(node.upper)
            v = node.variable
            if upper:
                return f"∑ {v} in Finset.range ({upper}), {body}"
            return f"∑ {v}, {body}"

        if isinstance(node, ProductNode):
            body = self._expr(node.body)
            upper = self._bound(node.upper)
            v = node.variable
            if upper:
                return f"∏ {v} in Finset.range ({upper}), {body}"
            return f"∏ {v}, {body}"

        if isinstance(node, IntegralNode):
            body = self._expr(node.body)
            lower = self._bound(node.lower)
            upper = self._bound(node.upper)
            v = node.variable
            if lower and upper:
                return f"∫ {v} in {lower}..{upper}, {body}"
            return f"∫ {v}, {body}"

        if isinstance(node, QuantifierNode):
            dom = (
                f" : {self._expr(node.domain)}"
                if node.domain else ""
            )
            body = self._expr(node.body)
            if node.kind == "forall":
                return f"∀ ({node.variable}{dom}), {body}"
            else:
                return f"∃ {node.variable}{dom}, {body}"

        if isinstance(node, LogicBinNode):
            left = self._expr(node.left)
            right = self._expr(node.right)
            return f"({left} {node.op} {right})"

        if isinstance(node, SetMemberNode):
            elem = self._expr(node.element)
            dom = self._expr(node.domain)
            return f"{elem} ∈ {dom}"

        if isinstance(node, SubsetNode):
            return f"{self._expr(node.left)} ⊆ {self._expr(node.right)}"

        if isinstance(node, SetOpNode):
            return (
                f"{self._expr(node.left)} "
                f"{node.op} "
                f"{self._expr(node.right)}"
            )

        # fallback
        return repr(node)

    def _bound(self, node: Optional[MathNode]) -> str:
        if node is None:
            return ""
        # for sum/prod lower bounds like i=0, extract the 0
        if isinstance(node, BinaryOpNode) and node.op == "=":
            return self._expr(node.right)
        return self._expr(node)

    def _paren_if_needed(self, node: MathNode, parent_op: str) -> str:
        """Wrap child in parentheses when operator precedence requires it."""
        _prec = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
        child_str = self._expr(node)
        if isinstance(node, BinaryOpNode):
            if _prec.get(node.op, 0) < _prec.get(parent_op, 0):
                return f"({child_str})"
        return child_str
