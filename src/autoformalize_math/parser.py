"""
LaTeXParser
===========
Rule-based parser that converts LaTeX math strings into an AST.

Supported constructs
--------------------
Arithmetic:
    \\frac{a}{b}          → FractionNode
    \\sum_{i=0}^{n} expr  → SumNode
    \\prod_{i=1}^{n} expr → ProductNode
    \\int_{a}^{b} expr    → IntegralNode
    x^2, x_n             → BinaryOpNode('^') / VariableNode with subscript
    +, -, *, /           → BinaryOpNode

Logic:
    \\forall x \\in S, P  → QuantifierNode
    \\exists x, P         → QuantifierNode
    P \\implies Q         → LogicBinNode('→')
    P \\iff Q             → LogicBinNode('↔')
    P \\land Q            → LogicBinNode('∧')
    P \\lor Q             → LogicBinNode('∨')
    \\neg P               → NotNode

Sets:
    x \\in S              → SetMemberNode
    A \\subseteq B        → SubsetNode
    A \\cup B             → SetOpNode('∪')
    A \\cap B             → SetOpNode('∩')

Proof blocks:
    Structured proof text with "Assume", "Let", "Then", "Therefore" keywords
    → ProofNode with premises, steps, and conclusion

Design notes
------------
The parser is intentionally lenient: unrecognized tokens become VariableNodes.
This mirrors real-world mathematical text where notation is highly variable.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .ast_nodes import (
    MathNode, NumberNode, VariableNode, SymbolNode,
    BinaryOpNode, FractionNode, SumNode, ProductNode, IntegralNode, NegateNode,
    LogicBinNode, NotNode, QuantifierNode, SetMemberNode, SubsetNode, SetOpNode,
    ProofStep, ProofNode,
)

# ──────────────────────────────────────────────────────────────────────────────
# Token patterns (order matters: most specific first)
# ──────────────────────────────────────────────────────────────────────────────

_TOKENS: List[Tuple[str, str]] = [
    ("FRAC",        r"\\frac"),
    ("SUM",         r"\\sum"),
    ("PROD",        r"\\prod"),
    ("INT",         r"\\int"),
    ("FORALL",      r"\\forall"),
    ("EXISTS",      r"\\exists"),
    ("IMPLIES",     r"\\(?:implies|Rightarrow)"),
    ("IFF",         r"\\(?:iff|Leftrightarrow)"),
    ("LAND",        r"\\land"),
    ("LOR",         r"\\lor"),
    ("NEG",         r"\\neg"),
    ("IN",          r"\\in"),
    ("NOTIN",       r"\\notin"),
    ("SUBSETEQ",    r"\\subseteq"),
    ("SUBSET",      r"\\subset"),
    ("CUP",         r"\\cup"),
    ("CAP",         r"\\cap"),
    ("SETMINUS",    r"\\setminus"),
    ("SQRT",        r"\\sqrt"),
    ("CDOT",        r"\\cdot"),
    ("TIMES",       r"\\times"),
    ("INFTY",       r"\\infty"),
    ("MATHBB",      r"\\mathbb"),
    ("TEXT",        r"\\text\{[^}]*\}"),
    ("COMMAND",     r"\\[a-zA-Z]+"),
    ("LBRACE",      r"\{"),
    ("RBRACE",      r"\}"),
    ("LBRACKET",    r"\["),
    ("RBRACKET",    r"\]"),
    ("LPAREN",      r"\("),
    ("RPAREN",      r"\)"),
    ("CARET",       r"\^"),
    ("UNDERSCORE",  r"_"),
    ("COMMA",       r","),
    ("SEMICOLON",   r";"),
    ("DOT",         r"\."),
    ("DOTS",        r"\.\.\."),
    ("EQ",          r"="),
    ("NEQ",         r"\\neq|≠"),
    ("LEQ",         r"\\leq|≤"),
    ("GEQ",         r"\\geq|≥"),
    ("LT",          r"<"),
    ("GT",          r">"),
    ("PLUS",        r"\+"),
    ("MINUS",       r"-"),
    ("STAR",        r"\*"),
    ("SLASH",       r"/"),
    ("NUMBER",      r"\d+(?:\.\d+)?"),
    ("IDENT",       r"[a-zA-Z_][a-zA-Z0-9_']*"),
    ("SPACE",       r"\s+"),
    ("OTHER",       r"."),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKENS)
)

# Set / number-theory names that map to SymbolNode
_KNOWN_SYMBOLS = {
    "N", "Z", "Q", "R", "C",   # common sets as single letters
    "mathbb",
}

_LOGIC_KEYWORDS = {
    r"\implies", r"\Rightarrow",
    r"\iff", r"\Leftrightarrow",
    r"\land", r"\lor",
}


class Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r})"


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        value = m.group()
        if kind == "SPACE":
            continue
        tokens.append(Token(kind, value))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

class _Parser:
    """Recursive-descent parser over a flat token list."""

    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    # ── helpers ──

    def _peek(self, offset: int = 0) -> Optional[Token]:
        i = self._pos + offset
        if i < len(self._tokens):
            return self._tokens[i]
        return None

    def _consume(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            raise SyntaxError(
                f"Expected {kind!r}, got {tok!r} at position {self._pos}"
            )
        return self._consume()

    def _match(self, *kinds: str) -> Optional[Token]:
        tok = self._peek()
        if tok and tok.kind in kinds:
            return self._consume()
        return None

    # ── curly-brace group ──

    def _braced_group(self) -> MathNode:
        """Parse {expr} and return the inner expression."""
        if self._peek() and self._peek().kind == "LBRACE":
            self._consume()  # {
            node = self._parse_expr()
            # consume closing brace if present
            if self._peek() and self._peek().kind == "RBRACE":
                self._consume()
            return node
        # no braces — fall back to atom
        return self._parse_atom()

    # ── subscript/superscript helpers ──

    def _parse_sub_super(self) -> Tuple[Optional[MathNode], Optional[MathNode]]:
        """
        Consume optional _expr ^expr or ^expr _expr in any order.
        Returns (lower, upper).
        """
        lower: Optional[MathNode] = None
        upper: Optional[MathNode] = None
        for _ in range(2):
            tok = self._peek()
            if tok and tok.kind == "UNDERSCORE" and lower is None:
                self._consume()
                lower = self._braced_group()
            elif tok and tok.kind == "CARET" and upper is None:
                self._consume()
                upper = self._braced_group()
            else:
                break
        return lower, upper

    # ── extract variable name from subscript block ──

    @staticmethod
    def _var_from_node(node: Optional[MathNode]) -> str:
        if node is None:
            return "i"
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, BinaryOpNode) and node.op == "=":
            if isinstance(node.left, VariableNode):
                return node.left.name
        return "i"

    # ────────────────────────────── expression grammar ────────────────────────

    def _parse_expr(self) -> MathNode:
        """Top level: handles logic binary operators."""
        left = self._parse_additive()

        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "IMPLIES":
                self._consume()
                right = self._parse_additive()
                left = LogicBinNode("→", left, right)
            elif tok.kind == "IFF":
                self._consume()
                right = self._parse_additive()
                left = LogicBinNode("↔", left, right)
            elif tok.kind == "LAND":
                self._consume()
                right = self._parse_additive()
                left = LogicBinNode("∧", left, right)
            elif tok.kind == "LOR":
                self._consume()
                right = self._parse_additive()
                left = LogicBinNode("∨", left, right)
            elif tok.kind in ("IN", "NOTIN"):
                op = tok.value
                self._consume()
                right = self._parse_additive()
                left = SetMemberNode(left, right)
            elif tok.kind == "SUBSETEQ":
                self._consume()
                right = self._parse_additive()
                left = SubsetNode(left, right)
            elif tok.kind == "SUBSET":
                self._consume()
                right = self._parse_additive()
                left = SubsetNode(left, right)
            elif tok.kind == "CUP":
                self._consume()
                right = self._parse_additive()
                left = SetOpNode("∪", left, right)
            elif tok.kind == "CAP":
                self._consume()
                right = self._parse_additive()
                left = SetOpNode("∩", left, right)
            elif tok.kind == "SETMINUS":
                self._consume()
                right = self._parse_additive()
                left = SetOpNode("∖", left, right)
            elif tok.kind in ("EQ", "NEQ", "LEQ", "GEQ", "LT", "GT"):
                op = tok.value
                self._consume()
                right = self._parse_additive()
                left = BinaryOpNode(op, left, right)
            else:
                break

        return left

    def _parse_additive(self) -> MathNode:
        left = self._parse_multiplicative()
        while True:
            tok = self._peek()
            if tok and tok.kind == "PLUS":
                self._consume()
                right = self._parse_multiplicative()
                left = BinaryOpNode("+", left, right)
            elif tok and tok.kind == "MINUS":
                self._consume()
                right = self._parse_multiplicative()
                left = BinaryOpNode("-", left, right)
            else:
                break
        return left

    def _parse_multiplicative(self) -> MathNode:
        left = self._parse_power()
        while True:
            tok = self._peek()
            if tok and tok.kind in ("STAR", "CDOT", "TIMES"):
                self._consume()
                right = self._parse_power()
                left = BinaryOpNode("*", left, right)
            elif tok and tok.kind == "SLASH":
                self._consume()
                right = self._parse_power()
                left = BinaryOpNode("/", left, right)
            else:
                break
        return left

    def _parse_power(self) -> MathNode:
        base = self._parse_unary()
        tok = self._peek()
        if tok and tok.kind == "CARET":
            self._consume()
            exp = self._braced_group()
            return BinaryOpNode("^", base, exp)
        return base

    def _parse_unary(self) -> MathNode:
        tok = self._peek()
        if tok and tok.kind == "MINUS":
            self._consume()
            operand = self._parse_atom()
            return NegateNode(operand)
        if tok and tok.kind == "NEG":
            self._consume()
            operand = self._parse_atom()
            return NotNode(operand)
        return self._parse_atom()

    def _parse_atom(self) -> MathNode:  # noqa: C901 – inherently branchy
        tok = self._peek()
        if tok is None:
            return VariableNode("?")

        # ── \frac{a}{b} ──
        if tok.kind == "FRAC":
            self._consume()
            num = self._braced_group()
            den = self._braced_group()
            return FractionNode(num, den)

        # ── \sum_{i=0}^{n} body ──
        if tok.kind == "SUM":
            self._consume()
            lower, upper = self._parse_sub_super()
            body = self._parse_atom()
            var = self._var_from_node(lower)
            return SumNode(var, lower, upper, body)

        # ── \prod_{i=1}^{n} body ──
        if tok.kind == "PROD":
            self._consume()
            lower, upper = self._parse_sub_super()
            body = self._parse_atom()
            var = self._var_from_node(lower)
            return ProductNode(var, lower, upper, body)

        # ── \int_{a}^{b} body ──
        if tok.kind == "INT":
            self._consume()
            lower, upper = self._parse_sub_super()
            body = self._parse_atom()
            # try to extract d<var> from body
            var = "x"
            return IntegralNode(var, lower, upper, body)

        # ── \forall x \in S, body ──
        if tok.kind == "FORALL":
            self._consume()
            var_tok = self._peek()
            var_name = "x"
            if var_tok and var_tok.kind == "IDENT":
                var_name = var_tok.value
                self._consume()
            domain: Optional[MathNode] = None
            if self._peek() and self._peek().kind == "IN":
                self._consume()
                domain = self._parse_atom()
            # skip comma
            self._match("COMMA")
            body = self._parse_expr()
            return QuantifierNode("forall", var_name, domain, body)

        # ── \exists x, body ──
        if tok.kind == "EXISTS":
            self._consume()
            var_tok = self._peek()
            var_name = "x"
            if var_tok and var_tok.kind == "IDENT":
                var_name = var_tok.value
                self._consume()
            domain2: Optional[MathNode] = None
            if self._peek() and self._peek().kind == "IN":
                self._consume()
                domain2 = self._parse_atom()
            self._match("COMMA")
            body = self._parse_expr()
            return QuantifierNode("exists", var_name, domain2, body)

        # ── \neg expr ──
        if tok.kind == "NEG":
            self._consume()
            operand = self._parse_atom()
            return NotNode(operand)

        # ── \sqrt{x} ──
        if tok.kind == "SQRT":
            self._consume()
            arg = self._braced_group()
            # represent as x^(1/2)
            return BinaryOpNode(
                "^", arg, FractionNode(NumberNode("1"), NumberNode("2"))
            )

        # ── \mathbb{N} etc. ──
        if tok.kind == "MATHBB":
            self._consume()
            inner = self._braced_group()
            name = inner.name if isinstance(inner, VariableNode) else "?"
            return SymbolNode(f"𝕄{name}")

        # ── generic \command → SymbolNode ──
        if tok.kind == "COMMAND":
            self._consume()
            return SymbolNode(tok.value)

        # ── {…} group ──
        if tok.kind == "LBRACE":
            return self._braced_group()

        # ── (…) group ──
        if tok.kind == "LPAREN":
            self._consume()
            inner = self._parse_expr()
            self._match("RPAREN")
            return inner

        # ── numbers ──
        if tok.kind == "NUMBER":
            self._consume()
            return NumberNode(tok.value)

        # ── identifiers ──
        if tok.kind == "IDENT":
            self._consume()
            node: MathNode = VariableNode(tok.value)
            # handle subscript attached to variable, e.g. x_n
            if self._peek() and self._peek().kind == "UNDERSCORE":
                self._consume()
                sub = self._braced_group()
                sub_str = sub.name if isinstance(sub, VariableNode) else repr(sub)
                node = VariableNode(f"{tok.value}_{sub_str}")
            return node

        # ── skip unrecognised token ──
        self._consume()
        return VariableNode(tok.value)

    # ── entry point ──

    def parse(self) -> MathNode:
        node = self._parse_expr()
        return node


# ──────────────────────────────────────────────────────────────────────────────
# Proof-structure parser
# ──────────────────────────────────────────────────────────────────────────────

# Sentence patterns for informal proof text
_PREMISE_RE = re.compile(
    r"(?i)(?:assume|let|suppose|given|hypothesis)[:\s,]+(.+?)(?:\.|$)"
)
_STEP_RE = re.compile(
    r"(?i)(?:then|hence|thus|so|note|observe|we have|it follows that|since)[:\s,]+(.+?)(?:\.|$)"
)
_CONCLUSION_RE = re.compile(
    r"(?i)(?:therefore|consequently|we conclude|which proves|q\.e\.d\.?|qed|□|∎)[:\s,]*(.*)(?:\.|$)"
)
_INDUCTION_RE = re.compile(
    r"(?i)(?:base case|inductive step|induction hypothesis)"
)


def _extract_latex_blocks(text: str) -> List[str]:
    """Pull out $…$ and $$…$$ math from prose."""
    return re.findall(r"\$\$(.+?)\$\$|\$(.+?)\$", text, re.DOTALL)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class LaTeXParser:
    """
    Parse LaTeX mathematical expressions and proof text into an AST.

    Usage
    -----
    >>> parser = LaTeXParser()
    >>> ast = parser.parse_expression(r"\\frac{p}{q}")
    >>> proof = parser.parse_proof("irrationality_sqrt2", proof_text)
    """

    def parse_expression(self, latex: str) -> MathNode:
        """
        Parse a LaTeX math expression string and return its AST.

        Parameters
        ----------
        latex : str
            A raw LaTeX math string (no surrounding $ needed).

        Returns
        -------
        MathNode
            Root node of the expression AST.
        """
        # strip surrounding dollars if present
        latex = latex.strip()
        latex = re.sub(r"^\$\$?|\$\$?$", "", latex).strip()
        tokens = tokenize(latex)
        parser = _Parser(tokens)
        return parser.parse()

    def parse_proof(self, theorem_name: str, proof_text: str) -> ProofNode:
        """
        Parse an informal proof written in semi-formal LaTeX prose.

        The parser recognises sentences beginning with proof-keyword markers
        (Assume, Let, Then, Therefore, …) and extracts:
          - premises  – assumptions / hypotheses
          - steps     – intermediate deductive steps
          - conclusion – the final claim

        Inline math ($…$) within each sentence is parsed into AST nodes.

        Parameters
        ----------
        theorem_name : str
            Human-readable name for the theorem being proved.
        proof_text : str
            Free-form proof text, potentially containing LaTeX math.

        Returns
        -------
        ProofNode
            Structured proof AST.
        """
        proof = ProofNode(theorem_name=theorem_name)

        # split into sentences (crude but robust)
        sentences = re.split(r"(?<=[.!?])\s+", proof_text.strip())

        step_counter = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # parse any inline math in the sentence
            math_node = self._best_math_from_sentence(sentence)

            if _CONCLUSION_RE.search(sentence):
                proof.conclusion = math_node
            elif _PREMISE_RE.search(sentence):
                proof.premises.append(math_node)
            elif _STEP_RE.search(sentence) or _INDUCTION_RE.search(sentence):
                step_counter += 1
                label = f"step{step_counter}"
                just = _classify_justification(sentence)
                proof.steps.append(ProofStep(label, math_node, just))
            else:
                # default: treat as a proof step
                step_counter += 1
                label = f"step{step_counter}"
                proof.steps.append(ProofStep(label, math_node, ""))

        # if no explicit conclusion found, use last step
        if proof.conclusion is None and proof.steps:
            last = proof.steps[-1]
            proof.conclusion = last.statement

        return proof

    # ── internal helpers ──

    def _best_math_from_sentence(self, sentence: str) -> MathNode:
        """
        Extract the most significant math node from a sentence.
        Falls back to a VariableNode carrying the raw sentence text.
        """
        blocks = _extract_latex_blocks(sentence)
        if blocks:
            # take the longest math block
            candidates = [b[0] or b[1] for b in blocks]
            longest = max(candidates, key=len)
            return self.parse_expression(longest)
        # no inline math → try parsing the whole sentence minus stopwords
        clean = _strip_proof_keywords(sentence)
        if clean:
            try:
                return self.parse_expression(clean)
            except Exception:
                pass
        return VariableNode(sentence[:80])  # fallback: raw text as label


def _classify_justification(sentence: str) -> str:
    """Heuristically identify the proof technique in a sentence."""
    s = sentence.lower()
    if "induct" in s:
        return "induction"
    if "contradict" in s:
        return "contradiction"
    if "assume" in s or "suppose" in s:
        return "assumption"
    if "definition" in s or "by def" in s:
        return "definition"
    if "algebra" in s or "simplif" in s or "expand" in s:
        return "algebra"
    if "theorem" in s or "lemma" in s:
        return "theorem"
    return "deduction"


def _strip_proof_keywords(sentence: str) -> str:
    """Remove leading proof keywords to expose the math."""
    pattern = re.compile(
        r"(?i)^(?:assume|let|suppose|given|then|hence|thus|so|note|"
        r"therefore|consequently|we have|it follows that|since)[:\s,]*"
    )
    return pattern.sub("", sentence).strip()
