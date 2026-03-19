"""
AST node definitions for mathematical expressions parsed from LaTeX.

Each node represents a grammatical construct in mathematical writing:
  - atoms (numbers, variables, symbols)
  - operations (arithmetic, logic, calculus)
  - proof structure (premises, steps, conclusion)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any


# ─────────────────────────── base ────────────────────────────

@dataclass
class MathNode:
    """Base class for all AST nodes."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ─────────────────────────── atoms ───────────────────────────

@dataclass
class NumberNode(MathNode):
    value: str  # keeps original string so we don't lose precision

    def __repr__(self) -> str:
        return f"Number({self.value})"


@dataclass
class VariableNode(MathNode):
    name: str

    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass
class SymbolNode(MathNode):
    """Named mathematical constant or logical symbol (∀, ∃, ℕ, …)."""
    name: str

    def __repr__(self) -> str:
        return f"Symbol({self.name})"


# ─────────────────────────── arithmetic ──────────────────────

@dataclass
class BinaryOpNode(MathNode):
    op: str          # +, -, *, /, ^
    left: MathNode
    right: MathNode

    def __repr__(self) -> str:
        return f"BinOp({self.op}, {self.left!r}, {self.right!r})"


@dataclass
class FractionNode(MathNode):
    numerator: MathNode
    denominator: MathNode

    def __repr__(self) -> str:
        return f"Frac({self.numerator!r}, {self.denominator!r})"


@dataclass
class SumNode(MathNode):
    variable: str
    lower: Optional[MathNode]
    upper: Optional[MathNode]
    body: MathNode

    def __repr__(self) -> str:
        return f"Sum({self.variable}, {self.lower!r}..{self.upper!r}, {self.body!r})"


@dataclass
class ProductNode(MathNode):
    variable: str
    lower: Optional[MathNode]
    upper: Optional[MathNode]
    body: MathNode

    def __repr__(self) -> str:
        return f"Prod({self.variable}, {self.lower!r}..{self.upper!r}, {self.body!r})"


@dataclass
class IntegralNode(MathNode):
    variable: str
    lower: Optional[MathNode]
    upper: Optional[MathNode]
    body: MathNode

    def __repr__(self) -> str:
        return f"Integral({self.variable}, {self.lower!r}..{self.upper!r}, {self.body!r})"


@dataclass
class NegateNode(MathNode):
    operand: MathNode

    def __repr__(self) -> str:
        return f"Neg({self.operand!r})"


# ─────────────────────────── logic ───────────────────────────

@dataclass
class LogicBinNode(MathNode):
    op: str   # ∧, ∨, →, ↔
    left: MathNode
    right: MathNode

    def __repr__(self) -> str:
        return f"Logic({self.op}, {self.left!r}, {self.right!r})"


@dataclass
class NotNode(MathNode):
    operand: MathNode

    def __repr__(self) -> str:
        return f"Not({self.operand!r})"


@dataclass
class QuantifierNode(MathNode):
    kind: str      # forall | exists
    variable: str
    domain: Optional[MathNode]
    body: MathNode

    def __repr__(self) -> str:
        return f"Quant({self.kind}, {self.variable}, {self.body!r})"


@dataclass
class SetMemberNode(MathNode):
    element: MathNode
    domain: MathNode

    def __repr__(self) -> str:
        return f"Member({self.element!r} ∈ {self.domain!r})"


@dataclass
class SubsetNode(MathNode):
    left: MathNode
    right: MathNode

    def __repr__(self) -> str:
        return f"Subset({self.left!r} ⊆ {self.right!r})"


@dataclass
class SetOpNode(MathNode):
    op: str   # ∪, ∩, \\
    left: MathNode
    right: MathNode

    def __repr__(self) -> str:
        return f"SetOp({self.op}, {self.left!r}, {self.right!r})"


# ─────────────────────────── proof structure ─────────────────

@dataclass
class ProofStep(MathNode):
    label: str
    statement: MathNode
    justification: str = ""

    def __repr__(self) -> str:
        return f"Step({self.label}: {self.statement!r})"


@dataclass
class ProofNode(MathNode):
    theorem_name: str
    premises: List[MathNode] = field(default_factory=list)
    steps: List[ProofStep] = field(default_factory=list)
    conclusion: Optional[MathNode] = None

    def __repr__(self) -> str:
        return (
            f"Proof({self.theorem_name!r}, "
            f"premises={len(self.premises)}, "
            f"steps={len(self.steps)})"
        )
