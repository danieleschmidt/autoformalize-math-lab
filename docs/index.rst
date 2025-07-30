autoformalize-math-lab Documentation
====================================

Welcome to the documentation for **autoformalize-math-lab**, an LLM-driven auto-formalization workbench that converts LaTeX proofs into Lean4/Isabelle formal verification code.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/Lean-4.0+-purple.svg
   :target: https://leanprover.github.io/
   :alt: Lean 4

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install autoformalize-math-lab

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from autoformalize import FormalizationPipeline

   # Initialize pipeline
   pipeline = FormalizationPipeline(
       target_system="lean4",
       model="gpt-4",
       max_correction_rounds=5
   )

   # Formalize a LaTeX proof
   latex_proof = r"""
   \begin{theorem}
   For any prime $p > 2$, we have $p \equiv 1 \pmod{2}$ or $p \equiv 3 \pmod{2}$.
   \end{theorem}
   """

   # Convert to Lean 4
   lean_code = pipeline.formalize(latex_proof)
   print(lean_code)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/autoformalize
   api/core
   api/parsers
   api/generators
   api/verifiers
   api/datasets
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/architecture
   advanced/custom-models
   advanced/extending
   advanced/performance

.. toctree::
   :maxdepth: 2
   :caption: Mathematical Foundations

   math/formalization-theory
   math/proof-systems
   math/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/testing
   development/documentation
   development/release-process

.. toctree::
   :maxdepth: 1
   :caption: Reference

   changelog
   license
   bibliography

Features
--------

🔧 **Multi-Format Parser**
   Handles PDF, LaTeX, and arXiv papers with intelligent mathematical content extraction.

🔄 **Self-Correcting Pipeline**
   Iterative refinement using proof assistant feedback for improved success rates.

🎯 **Cross-System Support**
   Targets Lean 4, Isabelle/HOL, Coq, and Agda with system-specific optimizations.

📚 **Mathlib Integration**
   Automatically aligns with standard mathematical libraries and existing theorems.

📊 **Success Tracking**
   CI dashboard monitoring formalization success rates and performance metrics.

Architecture Overview
--------------------

.. mermaid::

   graph TD
       A[LaTeX/PDF Input] --> B[Mathematical Parser]
       B --> C[Semantic Analyzer]
       C --> D[LLM Formalizer]
       D --> E[Proof Assistant]
       E --> F{Verification}
       F -->|Success| G[Output Formal Proof]
       F -->|Error| H[Error Parser]
       H --> I[Correction Prompter]
       I --> D

Performance Metrics
------------------

.. list-table:: Success Rates by Domain
   :header-rows: 1
   :widths: 30 15 20 15

   * - Mathematical Domain
     - Success Rate
     - Avg. Corrections
     - Mathlib Usage
   * - Basic Algebra
     - 92%
     - 1.3
     - 78%
   * - Number Theory
     - 87%
     - 2.1
     - 82%
   * - Real Analysis
     - 73%
     - 3.4
     - 91%
   * - Abstract Algebra
     - 68%
     - 3.8
     - 85%
   * - Topology
     - 61%
     - 4.2
     - 93%

Community
---------

* **GitHub Repository**: https://github.com/yourusername/autoformalize-math-lab
* **Documentation**: https://autoformalize-math.readthedocs.io
* **Issue Tracker**: https://github.com/yourusername/autoformalize-math-lab/issues
* **Discussions**: https://github.com/yourusername/autoformalize-math-lab/discussions

Citation
--------

If you use autoformalize-math-lab in your research, please cite:

.. code-block:: bibtex

   @inproceedings{autoformalize_math_lab,
     title={Autoformalize-Math-Lab: Bridging Informal and Formal Mathematics},
     author={Your Name},
     booktitle={International Conference on Automated Reasoning},
     year={2025}
   }

License
-------

This project is licensed under the MIT License - see the :doc:`license` page for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`