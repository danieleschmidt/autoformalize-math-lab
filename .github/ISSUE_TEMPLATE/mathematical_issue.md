---
name: Mathematical Issue
about: Report issues with mathematical formalization, proofs, or theorem translation
title: '[MATH] '
labels: ['mathematics', 'formalization', 'needs-review']
assignees: []
---

## 🔢 Mathematical Issue Type

- [ ] Incorrect formalization
- [ ] Proof verification failure  
- [ ] Mathematical notation parsing error
- [ ] Logic translation issue
- [ ] Domain-specific problem
- [ ] Theorem library integration issue

## 📐 Mathematical Domain

- [ ] Algebra (Linear, Abstract, etc.)
- [ ] Analysis (Real, Complex, Functional)
- [ ] Topology (General, Algebraic)
- [ ] Number Theory
- [ ] Logic & Set Theory
- [ ] Geometry (Euclidean, Differential)
- [ ] Category Theory
- [ ] Probability & Statistics
- [ ] Other: ___________

## 📝 Problem Statement

**Original Mathematical Statement:**
```latex
\begin{theorem}
State the original theorem or mathematical statement here
\end{theorem}

\begin{proof}
Include the original proof if relevant
\end{proof}
```

**Current Formalization Attempt:**
```lean
-- Current Lean 4 code (or Isabelle, Coq, etc.)
theorem problematic_theorem : Prop := by
  sorry -- Include the current attempt
```

## ❌ Issue Description

Describe the specific mathematical issue:

- What mathematical concept is not being handled correctly?
- Is the formalization semantically incorrect?
- Are there missing dependencies or imports?
- Is the proof strategy inappropriate?

## ✅ Expected Mathematical Behavior

**Correct Formalization Should Be:**
```lean
-- What the correct formalization should look like
theorem correct_theorem : Prop := by
  -- Expected proof approach
  exact sorry
```

**Mathematical Justification:**
Explain why this is the correct mathematical approach.

## 🔍 Mathematical Analysis

**Proof Strategy:**
- What proof technique should be used? (induction, contradiction, direct proof, etc.)
- Are there key lemmas or theorems that should be referenced?
- What mathematical libraries or imports are needed?

**Formalization Challenges:**
- Are there definitional issues?
- Type theory complications?
- Notation conflicts?
- Axiom dependencies?

## 📚 References

**Mathematical Sources:**
- Textbook references (author, title, page/theorem number)
- Research papers (arXiv links, DOI)
- Online mathematical resources
- Related formalizations in Mathlib, Archive of Formal Proofs, etc.

**Existing Formalizations:**
- Link to similar theorems in Mathlib
- Related work in other proof assistants
- Mathematical library dependencies

## 🎯 Target System Details

**Primary Target:**
- [ ] Lean 4 (specify Mathlib version if relevant)
- [ ] Isabelle/HOL (specify version)
- [ ] Coq (specify libraries)
- [ ] Agda
- [ ] Other: ___________

**Required Libraries/Imports:**
```lean
-- List necessary imports
import Mathlib.Algebra.Basic
import Mathlib.Logic.Basic
-- etc.
```

## 🧪 Test Cases

**Minimal Example:**
```latex
% Simplest case that demonstrates the issue
\begin{lemma}
Simple case that fails
\end{lemma}
```

**Extended Examples:**
```latex
% More complex cases
\begin{theorem}
Related theorems that should also work
\end{theorem}
```

## 🔄 Reproduction Steps

1. Input the LaTeX mathematical statement
2. Run formalization with specific settings: [...]
3. Observe the error/incorrect output
4. Expected: correct formalization
5. Actual: [describe the problem]

## 💡 Suggested Solutions

**Mathematical Approach:**
- Alternative proof strategies
- Different formalization approaches  
- Required lemma developments

**Technical Implementation:**
- Parser improvements needed
- Generator modifications
- Verification enhancements

## 🎓 Mathematical Difficulty Level

- [ ] Undergraduate level
- [ ] Graduate level  
- [ ] Research level
- [ ] Elementary/High school level

## 👥 Expert Review Needed

- [ ] Requires domain expert review
- [ ] Needs formal verification expert
- [ ] Should be reviewed by Mathlib contributors
- [ ] Community discussion recommended

## 📊 Impact on Related Mathematics

**Affected Areas:**
- What other mathematical concepts depend on this?
- Are there downstream formalization impacts?
- Could this affect proof automation?

## 🏷️ Priority Level

- [ ] Critical - Fundamental mathematical error
- [ ] High - Important theorem or common pattern
- [ ] Medium - Specific use case or advanced topic
- [ ] Low - Edge case or rare scenario

## 📎 Additional Mathematical Context

Provide any additional mathematical background, intuition, or context that might help with the formalization.

## 🤝 Mathematical Collaboration

Are you able to assist with the mathematical aspects?

- [ ] I can provide mathematical guidance
- [ ] I can review proposed solutions
- [ ] I can test the formalization
- [ ] I can provide additional examples
- [ ] I need mathematical assistance to understand the issue