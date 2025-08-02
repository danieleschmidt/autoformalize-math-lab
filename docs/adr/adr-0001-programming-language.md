# ADR-0001: Choice of Programming Language

## Status
Accepted

## Context
We need to choose a primary programming language for the Autoformalize Math Lab project. The system needs to interface with multiple proof assistants, process mathematical content, and integrate with various LLM APIs.

Key requirements:
- Strong ecosystem for mathematical computing and NLP
- Good interoperability with proof assistants (Lean 4, Isabelle, Coq)
- Mature LLM integration libraries
- Rich testing and development tooling
- Active community and long-term support

## Decision
We will use Python 3.9+ as the primary programming language for the project.

## Consequences

### Positive
- **Rich Mathematical Ecosystem**: NumPy, SciPy, SymPy for mathematical computations
- **NLP Libraries**: NLTK, spaCy, transformers for text processing
- **LLM Integration**: Mature libraries for OpenAI, Anthropic, and other providers
- **Proof Assistant Integration**: Good subprocess/pexpect support for external tools
- **Development Velocity**: Rapid prototyping and iteration capabilities
- **Community**: Large community and extensive documentation
- **Testing**: Excellent testing frameworks (pytest, hypothesis)

### Negative
- **Performance**: May be slower than compiled languages for compute-intensive tasks
- **Deployment**: Requires Python runtime and dependency management
- **Type Safety**: Dynamic typing can lead to runtime errors (mitigated by mypy)

### Mitigations
- Use type hints and mypy for better type safety
- Optimize performance-critical paths with NumPy/SciPy
- Consider Rust/C++ extensions for future performance bottlenecks
- Use proper dependency management with pyproject.toml

## Alternative Considered
- **Rust**: Better performance but smaller ecosystem for mathematical/NLP tasks
- **TypeScript/Node.js**: Good for web integration but limited mathematical libraries
- **Julia**: Excellent for mathematical computing but less mature LLM ecosystem
- **Lean 4**: Native integration but limited general-purpose libraries

## References
- [Python for Scientific Computing](https://scipy.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Mathematical Python](https://www.math.ubc.ca/~pwalls/math-python/)