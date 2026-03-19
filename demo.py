#!/usr/bin/env python3
"""
demo.py
=======
Demonstrates the autoformalize-math-lab pipeline on three classical theorems:

  1. Irrationality of √2             (proof by contradiction)
  2. Sum of first n naturals          (proof by induction)
  3. De Morgan's Law for sets         (basic set theory)

Each proof is run through:
  LaTeXParser → FormalConverter → ProofChecker
and the results are printed to stdout.
"""

from autoformalize_math import LaTeXParser, FormalConverter, ProofChecker

parser = LaTeXParser()
converter = FormalConverter()
checker = ProofChecker()


def run_demo(theorem_name: str, proof_text: str) -> None:
    separator = "=" * 70
    print(separator)
    print(f"THEOREM: {theorem_name}")
    print(separator)

    # 1. Parse
    proof_ast = parser.parse_proof(theorem_name, proof_text)
    print(f"\n[AST]  {proof_ast!r}")
    print(f"  Premises : {len(proof_ast.premises)}")
    print(f"  Steps    : {len(proof_ast.steps)}")
    print(f"  Conclusion: {proof_ast.conclusion!r}")

    # 2. Convert
    lean_code = converter.convert_proof(proof_ast)
    print(f"\n[Lean4 Pseudocode]\n{lean_code}")

    # 3. Check
    report = checker.validate(proof_ast)
    print(f"\n[Validation]\n{report.summary()}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Proof 1 — Irrationality of √2
# Proof by contradiction: assume √2 = p/q in lowest terms,
# derive that both p and q are even → contradiction.
# ──────────────────────────────────────────────────────────────────────────────

SQRT2_PROOF = r"""
Assume $\sqrt{2} = \frac{p}{q}$ where $p, q \in \mathbb{Z}$ and $\gcd(p, q) = 1$.
Then $2 = \frac{p^2}{q^2}$, so $p^2 = 2 q^2$.
Hence $p^2$ is even, so $p$ is even.
Let $p = 2k$ for some integer $k$.
Then $4k^2 = 2q^2$, so $q^2 = 2k^2$.
Hence $q^2$ is even, so $q$ is even.
Therefore $\gcd(p, q) \geq 2$, contradicting $\gcd(p, q) = 1$.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Proof 2 — Sum of first n naturals (Gauss)
# Proof by induction: base case n=0, inductive step.
# ──────────────────────────────────────────────────────────────────────────────

GAUSS_SUM_PROOF = r"""
Assume $\sum_{i=0}^{n} i = \frac{n(n+1)}{2}$ holds for some $n \geq 0$.
Base case: $\sum_{i=0}^{0} i = 0 = \frac{0 \cdot 1}{2}$.
Inductive step: assume $\sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
Then $\sum_{i=0}^{n+1} i = \frac{n(n+1)}{2} + (n+1)$.
Hence $\sum_{i=0}^{n+1} i = \frac{n(n+1) + 2(n+1)}{2} = \frac{(n+1)(n+2)}{2}$.
Therefore $\forall n \in \mathbb{N}, \sum_{i=0}^{n} i = \frac{n(n+1)}{2}$.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Proof 3 — De Morgan's Law for sets: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
# ──────────────────────────────────────────────────────────────────────────────

DEMORGAN_PROOF = r"""
Assume $x \in (A \cup B)^c$.
Then $x \notin A \cup B$.
Hence $x \notin A$ and $x \notin B$.
Thus $x \in A^c$ and $x \in B^c$.
Therefore $x \in A^c \cap B^c$, so $(A \cup B)^c \subseteq A^c \cap B^c$.
Conversely, assume $x \in A^c \cap B^c$.
Then $x \notin A$ and $x \notin B$, hence $x \notin A \cup B$.
Therefore $x \in (A \cup B)^c$, proving $A^c \cap B^c \subseteq (A \cup B)^c$.
Therefore $(A \cup B)^c = A^c \cap B^c$.
"""

if __name__ == "__main__":
    run_demo("irrationality_sqrt2", SQRT2_PROOF)
    run_demo("sum_of_naturals_gauss", GAUSS_SUM_PROOF)
    run_demo("demorgan_set_law", DEMORGAN_PROOF)
