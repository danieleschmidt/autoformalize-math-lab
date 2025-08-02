"""Sample mathematical theorems for testing."""

from typing import Dict, Any

# Sample LaTeX theorems by mathematical domain
ALGEBRA_THEOREMS = {
    "quadratic_formula": {
        "latex": r"""
\begin{theorem}[Quadratic Formula]
For any quadratic equation $ax^2 + bx + c = 0$ where $a \neq 0$, the solutions are:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
\end{theorem}
\begin{proof}
Starting from $ax^2 + bx + c = 0$, we complete the square.
Dividing by $a$: $x^2 + \frac{b}{a}x + \frac{c}{a} = 0$
Adding $\left(\frac{b}{2a}\right)^2$ to both sides:
$x^2 + \frac{b}{a}x + \left(\frac{b}{2a}\right)^2 = \left(\frac{b}{2a}\right)^2 - \frac{c}{a}$
This gives us $\left(x + \frac{b}{2a}\right)^2 = \frac{b^2 - 4ac}{4a^2}$
Taking square roots: $x + \frac{b}{2a} = \pm\frac{\sqrt{b^2 - 4ac}}{2a}$
Therefore: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$
\end{proof}
""",
        "lean4": """
theorem quadratic_formula (a b c : ℝ) (ha : a ≠ 0) :
  ∀ x, a * x^2 + b * x + c = 0 ↔ 
  x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ 
  x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a) :=
by sorry
""",
        "domain": "algebra",
        "difficulty": "undergraduate"
    },
    
    "binomial_theorem": {
        "latex": r"""
\begin{theorem}[Binomial Theorem]
For any real numbers $x$ and $y$, and any non-negative integer $n$:
$$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$$
\end{theorem}
\begin{proof}
We prove this by induction on $n$.
Base case: For $n = 0$, $(x+y)^0 = 1 = \binom{0}{0}x^0y^0$.
Inductive step: Assume the formula holds for $n$. Then:
$(x+y)^{n+1} = (x+y)(x+y)^n = (x+y)\sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$
$= \sum_{k=0}^{n} \binom{n}{k} x^{n+1-k} y^k + \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^{k+1}$
Using the binomial coefficient identity $\binom{n}{k} + \binom{n}{k-1} = \binom{n+1}{k}$,
we get $(x+y)^{n+1} = \sum_{k=0}^{n+1} \binom{n+1}{k} x^{n+1-k} y^k$.
\end{proof}
""",
        "lean4": """
theorem binomial_theorem (x y : ℝ) (n : ℕ) :
  (x + y) ^ n = ∑ k in Finset.range (n + 1), (n.choose k : ℝ) * x ^ (n - k) * y ^ k :=
by sorry
""",
        "domain": "algebra",
        "difficulty": "undergraduate"
    }
}

NUMBER_THEORY_THEOREMS = {
    "fundamental_theorem_arithmetic": {
        "latex": r"""
\begin{theorem}[Fundamental Theorem of Arithmetic]
Every integer greater than 1 is either prime or can be represented as a unique product of prime numbers.
\end{theorem}
\begin{proof}
We prove existence and uniqueness separately.
Existence: Let $n > 1$. If $n$ is prime, we are done. Otherwise, $n = ab$ where $1 < a, b < n$.
By strong induction, both $a$ and $b$ can be written as products of primes, so $n$ can too.
Uniqueness: Suppose $n = p_1 p_2 \cdots p_r = q_1 q_2 \cdots q_s$ where all factors are prime.
Since $p_1$ divides the right side and all $q_i$ are prime, $p_1 = q_j$ for some $j$.
Reordering and canceling, we get the same factorization by induction.
\end{proof}
""",
        "lean4": """
theorem fundamental_theorem_arithmetic (n : ℕ) (hn : n > 1) :
  ∃! (factors : Multiset ℕ), factors.toList.all Nat.Prime ∧ factors.prod = n :=
by sorry
""",
        "domain": "number_theory",
        "difficulty": "undergraduate"
    },
    
    "euclid_infinitude_primes": {
        "latex": r"""
\begin{theorem}[Euclid's Theorem]
There are infinitely many prime numbers.
\end{theorem}
\begin{proof}
Suppose there are only finitely many primes $p_1, p_2, \ldots, p_n$.
Consider $N = p_1 p_2 \cdots p_n + 1$.
Since $N > 1$, it has a prime divisor $p$.
But $p$ cannot be any of $p_1, \ldots, p_n$ since $N \equiv 1 \pmod{p_i}$ for all $i$.
Therefore $p$ is a new prime, contradicting our assumption.
\end{proof}
""",
        "lean4": """
theorem euclid_infinitude_primes : ∀ n : ℕ, ∃ p > n, Nat.Prime p :=
by sorry
""",
        "domain": "number_theory", 
        "difficulty": "undergraduate"
    }
}

ANALYSIS_THEOREMS = {
    "intermediate_value_theorem": {
        "latex": r"""
\begin{theorem}[Intermediate Value Theorem]
Let $f : [a, b] \to \mathbb{R}$ be continuous on $[a, b]$ with $f(a) \neq f(b)$.
For any value $y$ between $f(a)$ and $f(b)$, there exists $c \in (a, b)$ such that $f(c) = y$.
\end{theorem}
\begin{proof}
Without loss of generality, assume $f(a) < y < f(b)$.
Let $S = \{x \in [a, b] : f(x) \leq y\}$.
Since $a \in S$, we have $S \neq \emptyset$, and $S$ is bounded above by $b$.
Let $c = \sup S$. We claim $f(c) = y$.
If $f(c) < y$, then by continuity, there exists $\delta > 0$ such that 
$f(x) < y$ for all $x \in (c - \delta, c + \delta)$, contradicting $c = \sup S$.
If $f(c) > y$, then by continuity, there exists $\delta > 0$ such that
$f(x) > y$ for all $x \in (c - \delta, c + \delta)$, contradicting the definition of $c$.
Therefore $f(c) = y$.
\end{proof}
""",
        "lean4": """
theorem intermediate_value_theorem {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b)) (y : ℝ) 
  (hy : y ∈ Set.Ioo (f a) (f b) ∪ Set.Ioo (f b) (f a)) :
  ∃ c ∈ Set.Ioo a b, f c = y :=
by sorry
""",
        "domain": "analysis",
        "difficulty": "undergraduate"
    }
}

# All theorems combined
ALL_THEOREMS = {
    **ALGEBRA_THEOREMS,
    **NUMBER_THEORY_THEOREMS, 
    **ANALYSIS_THEOREMS
}

def get_theorem_by_domain(domain: str) -> Dict[str, Any]:
    """Get all theorems for a specific mathematical domain."""
    return {k: v for k, v in ALL_THEOREMS.items() if v["domain"] == domain}

def get_theorem_by_difficulty(difficulty: str) -> Dict[str, Any]:
    """Get all theorems of a specific difficulty level.""" 
    return {k: v for k, v in ALL_THEOREMS.items() if v["difficulty"] == difficulty}

def get_sample_latex_errors() -> Dict[str, str]:
    """Common LaTeX parsing errors for testing error handling."""
    return {
        "unmatched_braces": r"\begin{theorem} Missing closing brace",
        "unknown_command": r"\begin{theorem} \unknowncommand{test} \end{theorem}",
        "malformed_math": r"\begin{theorem} $x = y + \end{theorem}",
        "nested_environments": r"\begin{theorem} \begin{proof} \end{theorem} \end{proof}",
        "invalid_characters": r"\begin{theorem} Contains invalid chars: àáâ \end{theorem}"
    }

def get_sample_proof_assistant_errors() -> Dict[str, Dict[str, str]]:
    """Sample error messages from proof assistants."""
    return {
        "lean4": {
            "syntax_error": "syntax error: expected expression",
            "type_mismatch": "type mismatch: expected ℕ, got ℝ",  
            "unknown_identifier": "unknown identifier 'unknownTheorem'",
            "tactic_failed": "tactic 'simp' failed"
        },
        "isabelle": {
            "syntax_error": "Syntax error in line 5",
            "type_error": "Type unification failed",
            "unknown_constant": "Unknown constant: unknownConst",
            "proof_failed": "Failed to apply proof method"
        },
        "coq": {
            "syntax_error": "Syntax error: Vernacular entry expected",
            "type_error": "The term has type nat but is expected to have type bool",
            "unknown_identifier": "The reference unknownRef was not found",
            "tactic_error": "Tactic failure"
        }
    }