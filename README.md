# autoformalize-math-lab

> **Auto-formalization:** translating informal mathematical proofs written in LaTeX into Lean4-style verified pseudocode using rule-based pattern matching.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem: The Formalization Gap

Mathematics has a dirty secret. The proofs in textbooks, papers, and PhD theses are written in a semi-formal mix of natural language and LaTeX notation. Mathematicians understand them through shared convention, intuition, and years of training. But computers don't.

Formal proof assistants — **Lean4**, **Isabelle**, **Coq** — can verify proofs with mathematical certainty: every logical step is checked by a type checker with no room for hand-waving. The problem? Writing proofs in these systems takes roughly **10–100× longer** than writing them informally. A 2-page textbook proof can require 200+ lines of Lean4 code.

This creates a massive bottleneck. The mathematical knowledge encoded in millions of academic papers is effectively **inaccessible to formal verification tools**. We can't run Lean4 on a PDF.

**Auto-formalization** is the research problem of bridging this gap: given informal math, produce verified formal code automatically.

This is currently one of the hardest open problems in AI for mathematics. State-of-the-art LLMs (GPT-4, Gemini, Claude) can handle simple lemmas but fail on novel or complex proofs. The challenge sits at the intersection of:

- Natural language understanding (parsing mathematical prose)
- Symbolic reasoning (understanding proof structure)
- Type theory (generating well-typed Lean4 terms)
- Automated theorem proving (verifying the result)

Research venues targeting this: **ICML**, **NeurIPS**, **LICS**, **ITP**, **JAR**.

---

## What This Repo Does

This lab demonstrates the **structural decomposition** approach to auto-formalization without relying on LLM API calls. It uses rule-based pattern matching to show the core pipeline concept clearly:

```
LaTeX proof text
      ↓
  LaTeXParser         ← tokenizer + recursive-descent parser
      ↓
  Abstract Syntax Tree (AST)
      ↓
  FormalConverter     ← AST → Lean4-style pseudocode
      ↓
  Lean4 pseudocode
      ↓
  ProofChecker        ← structural validation
      ↓
  Validation report
```

**No LLM, no external dependencies** — pure Python 3.10+ standard library.

---

## Architecture

### `LaTeXParser`

Tokenizes LaTeX math strings and parses them into an AST using a recursive-descent parser.

Supports:
| Construct | LaTeX | AST Node |
|-----------|-------|----------|
| Fractions | `\frac{a}{b}` | `FractionNode` |
| Sums | `\sum_{i=0}^{n} f(i)` | `SumNode` |
| Products | `\prod_{k=1}^{n} k` | `ProductNode` |
| Integrals | `\int_{a}^{b} f(x)` | `IntegralNode` |
| Universal | `\forall x \in S, P` | `QuantifierNode` |
| Existential | `\exists x, P` | `QuantifierNode` |
| Implication | `P \implies Q` | `LogicBinNode('→')` |
| Iff | `P \iff Q` | `LogicBinNode('↔')` |
| Conjunction | `A \land B` | `LogicBinNode('∧')` |
| Disjunction | `A \lor B` | `LogicBinNode('∨')` |
| Negation | `\neg P` | `NotNode` |
| Set membership | `x \in S` | `SetMemberNode` |
| Subset | `A \subseteq B` | `SubsetNode` |
| Set union | `A \cup B` | `SetOpNode('∪')` |
| Set intersection | `A \cap B` | `SetOpNode('∩')` |
| Arithmetic | `+, -, *, /, ^` | `BinaryOpNode` |

The `parse_proof` method additionally extracts proof structure from informal text by recognising sentence-level markers:
- **Premises**: sentences starting with `Assume`, `Let`, `Suppose`, `Given`
- **Steps**: sentences starting with `Then`, `Hence`, `Thus`, `So`, `We have`
- **Conclusion**: sentences starting with `Therefore`, `Consequently`, `QED`, `□`

### `FormalConverter`

Converts AST nodes to Lean4-style pseudocode strings following Lean4 syntax conventions:

- `∀ (x : α), P x` — universal quantification with type annotation
- `∃ x, P x` — existential quantification
- `∑ i in Finset.range (n), f i` — finite sums
- `∏ i in Finset.range (n), f i` — finite products
- `∫ x in a..b, f x` — integrals
- `(p / q : ℚ)` — rational fractions with type coercion
- `ℕ ℤ ℚ ℝ ℂ` — Unicode number system symbols

Full proofs are rendered as:
```lean4
theorem <name> (h1 : <premise1>) (h2 : <premise2>) : <conclusion> := by
  have step1 : <statement> := by  -- <justification>
    exact?
  ...
  show <conclusion>
  exact?
```

### `ProofChecker`

Validates structural properties of a parsed proof:

| Check | Description |
|-------|-------------|
| `has_premises` | At least one assumption/hypothesis extracted |
| `has_conclusion` | A conclusion node was identified |
| `has_steps` | Proof contains intermediate deductive steps |
| `steps_non_trivial` | ≥50% of steps contain parsed math (not raw text) |
| `conclusion_follows` | Conclusion is structurally consistent with the proof body |
| `variable_scope` | Multi-character variables in conclusion are scoped by premises |

---

## Installation

```bash
git clone https://github.com/danieleschmidt/autoformalize-math-lab
cd autoformalize-math-lab
pip install -e .
```

No external dependencies. Requires Python 3.10+.

---

## Usage

### Run the demo

```bash
python demo.py
```

This processes three classical theorems through the full pipeline:

1. **Irrationality of √2** — proof by contradiction
2. **Gauss sum formula** — proof by induction
3. **De Morgan's Law for sets** — set theory proof

### Use the library

```python
from autoformalize_math import LaTeXParser, FormalConverter, ProofChecker

parser = LaTeXParser()
converter = FormalConverter()
checker = ProofChecker()

# Parse a single expression
node = parser.parse_expression(r"\forall n \in \mathbb{N}, \sum_{i=0}^{n} i = \frac{n(n+1)}{2}")
lean = converter.convert_expression(node)
print(lean)
# → ∀ (n : ℕ), ∑ i in Finset.range (n), i = (n(n + 1) / 2 : ℚ)

# Parse a full proof
proof_text = r"""
Assume $p, q \in \mathbb{Z}$ and $\gcd(p, q) = 1$.
Then $p^2 = 2q^2$.
Hence $p$ is even.
Therefore $\gcd(p, q) \geq 2$, a contradiction.
"""
proof = parser.parse_proof("irrationality_sqrt2", proof_text)
lean_code = converter.convert_proof(proof)
report = checker.validate(proof)

print(lean_code)
print(report.summary())
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

64 tests covering:
- All parser constructs (fractions, sums, products, integrals, logic, sets, arithmetic)
- Proof-structure extraction
- Expression-to-Lean4 conversion for all node types
- All ProofChecker structural validations
- End-to-end pipeline integration

---

## Example Output

**Input (informal LaTeX proof of √2 irrationality):**

```latex
Assume $\sqrt{2} = \frac{p}{q}$ where $p, q \in \mathbb{Z}$ and $\gcd(p, q) = 1$.
Then $p^2 = 2 q^2$.
Hence $p$ is even.
Let $p = 2k$ for some integer $k$.
Then $4k^2 = 2q^2$, so $q^2 = 2k^2$.
Hence $q$ is even.
Therefore $\gcd(p, q) \geq 2$, contradicting $\gcd(p, q) = 1$.
```

**Output (Lean4 pseudocode):**

```lean4
theorem irrationality_sqrt2 (h1 : 2 ^ (1 / 2 : ℚ) = (p / q : ℚ)) (h2 : p = 2) : \gcd := by
  have step1 : 2 = (p ^ 2 / q ^ 2 : ℚ) := by  -- deduction
    exact?
  have step2 : p ^ 2 := by  -- deduction
    exact?
  have step3 : 4 := by  -- deduction
    exact?
  have step4 : q ^ 2 := by  -- deduction
    exact?
  show \gcd
  exact?
```

**Validation report:**
```
Validation report for 'irrationality_sqrt2'
  Result: VALID  (6/6 checks passed)

  [✓] has_premises: 2 premise(s) found
  [✓] has_conclusion: conclusion is SymbolNode
  [✓] has_steps: 4 step(s) found
  [✓] steps_non_trivial: 4/4 steps contain parsed math (ratio 100%)
  [✓] conclusion_follows: conclusion is structurally consistent with proof body
  [✓] variable_scope: all multi-character conclusion variables are properly scoped
```

---

## Research Context

### Why this matters

The **formalization gap** is one of the most consequential unsolved problems in mathematics and AI:

- **Mathematics is accreting debt**: each year, thousands of papers contain proofs that have never been formally verified. Subtle errors propagate undetected for decades.
- **AI math reasoning is brittle**: LLMs can produce plausible-looking proofs that contain logical holes, because they optimize for human-readable output, not logical correctness.
- **Formal verification is underutilized**: Lean4's Mathlib library has formalized ~200,000 results — an impressive achievement, but a fraction of published mathematics.

Auto-formalization could change this by making formal verification accessible at scale.

### Current state of the art

- **Draft, Sketch, and Prove** (Jiang et al., 2022): LLM generates informal proof sketch, automated prover fills in the details
- **Hypertree Proof Search** (Lample et al., 2022): tree search over Lean tactics guided by learned value function
- **Lean Copilot** (Han et al., 2024): LLM-as-tactic-suggester within the Lean4 proof assistant
- **MathLib4**: massive human-curated formal library — the ground truth target for auto-formalization

### The rule-based approach (this repo)

Before throwing LLMs at the problem, it helps to understand the *structural* challenge. This repo answers the question: **how far can explicit pattern matching take us?**

The answer: far enough to demonstrate the pipeline and identify where the hard problems live:

1. **Ambiguous notation**: `A^c` could mean complement, power, or transpose depending on context
2. **Implicit quantifiers**: "for all n" is often left out when it's obvious to a human
3. **Proof gaps**: informal proofs routinely skip "obvious" steps that require dozens of Lean4 lines
4. **Semantic alignment**: mapping informal terms to Mathlib lemma names requires world knowledge

A production system needs to solve all of these — which is why this is a hard research problem.

### Future directions

- **LLM-guided parsing**: use an LLM to resolve ambiguity in the AST (which `^c` means complement here?)
- **Tactic generation**: map proof steps to Lean4 tactics (`ring`, `linarith`, `simp`, `omega`)
- **Mathlib alignment**: retrieve relevant Mathlib lemmas for each step using embedding search
- **Feedback loop**: use Lean4's type checker as a training signal for improving generation

---

## Project Structure

```
autoformalize-math-lab/
├── src/
│   └── autoformalize_math/
│       ├── __init__.py          # Public API
│       ├── ast_nodes.py         # All AST node dataclasses
│       ├── parser.py            # LaTeXParser (tokenizer + recursive descent)
│       ├── converter.py         # FormalConverter (AST → Lean4 pseudocode)
│       └── checker.py           # ProofChecker (structural validation)
├── tests/
│   ├── test_parser.py           # Parser unit tests
│   ├── test_converter.py        # Converter unit tests
│   └── test_checker.py          # Checker unit tests + end-to-end
├── demo.py                      # 3-proof demonstration
├── pyproject.toml
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
