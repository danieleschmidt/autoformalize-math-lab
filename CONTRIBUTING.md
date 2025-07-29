# Contributing to autoformalize-math-lab

Thank you for your interest in contributing to autoformalize-math-lab! This project aims to bridge the gap between informal mathematical writing and formal verification systems through automated translation of LaTeX proofs into formal proof assistant code.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Types of Contributions](#types-of-contributions)
- [Development Workflow](#development-workflow)
- [Mathematical Standards](#mathematical-standards)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submission Guidelines](#submission-guidelines)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- At least one proof assistant (Lean 4, Isabelle/HOL, or Coq)
- LaTeX distribution (for processing mathematical content)
- API access to an LLM service (OpenAI GPT-4, Claude, etc.)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/autoformalize-math-lab.git
   cd autoformalize-math-lab
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

6. **Install proof assistants** (choose one or more)
   ```bash
   # Lean 4
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   
   # Isabelle (download from https://isabelle.in.tum.de/)
   # Coq (via package manager or opam)
   ```

7. **Run tests to verify setup**
   ```bash
   make test
   ```

## Types of Contributions

### 🔬 Mathematical Formalization
- Adding support for new mathematical domains
- Improving proof translation accuracy
- Contributing domain-specific prompt templates
- Adding mathematical benchmark problems

### 🛠️ Technical Development
- Bug fixes and performance improvements
- New proof assistant backends
- Enhanced parsing capabilities
- Infrastructure and tooling improvements

### 📚 Documentation and Examples
- Tutorial improvements
- Example formalizations
- API documentation
- User guides and best practices

### 🧪 Testing and Quality Assurance
- Test case contributions
- Benchmark problem sets
- Error analysis and reporting
- Performance profiling

## Development Workflow

### Branch Naming

- `feature/feature-name` - New features
- `fix/issue-description` - Bug fixes
- `docs/update-description` - Documentation updates
- `test/test-description` - Test improvements
- `refactor/component-name` - Code refactoring

### Commit Messages

Use conventional commit format:

```
type(scope): brief description

Optional longer description explaining the change.

Closes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
```
feat(lean): add support for category theory translations
fix(parser): handle multi-line LaTeX theorem statements
docs(api): update formalization pipeline documentation
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   make test
   make lint
   make type-check
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat(component): description of changes"
   git push origin feature/your-feature-name
   ```

5. **Create a pull request**
   - Use the pull request template
   - Link to related issues
   - Include mathematical examples where relevant
   - Add reviewers familiar with the affected components

## Mathematical Standards

### Proof Accuracy
- All generated formal proofs must be verifiable by their target proof assistant
- Mathematical reasoning must be sound and complete
- Cite relevant mathematical literature and theorems

### Mathematical Notation
- Follow standard mathematical conventions
- Use Unicode symbols consistently
- Provide LaTeX source for complex mathematical expressions
- Document any non-standard notation

### Formalization Quality
- Prefer existing library theorems over custom definitions
- Use idiomatic proof assistant style
- Include clear mathematical comments
- Structure proofs in a readable, hierarchical manner

### Benchmark Standards
- Include both positive and negative test cases
- Cover edge cases and boundary conditions
- Provide expected outcomes and success metrics
- Include timing and performance expectations

## Code Standards

### Python Style
- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black formatter)
- Use docstrings for all public functions and classes

### Code Organization
```
src/autoformalize/
├── core/           # Core formalization logic
├── parsers/        # LaTeX and mathematical content parsers
├── generators/     # Proof assistant code generators
├── verifiers/      # Proof verification interfaces
├── models/         # LLM integration and prompting
├── utils/          # Utility functions
└── datasets/       # Benchmark datasets and evaluation
```

### Error Handling
- Use custom exception classes for domain-specific errors
- Provide informative error messages
- Log errors appropriately for debugging
- Handle proof assistant timeouts gracefully

### Logging
- Use structured logging with appropriate levels
- Include mathematical context in log messages
- Avoid logging sensitive information (API keys, etc.)

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for workflows
├── mathematical/   # Mathematical correctness tests
├── performance/    # Performance and benchmark tests
└── fixtures/       # Test data and examples
```

### Test Requirements
- Unit tests for all public functions
- Integration tests for complete formalization workflows
- Mathematical correctness verification
- Performance regression tests for critical paths

### Test Data
- Include diverse mathematical domains
- Cover various LaTeX formatting styles
- Test with different proof assistant versions
- Include both simple and complex mathematical content

### Running Tests
```bash
# All tests
make test

# Specific test categories
make test-unit
make test-integration
make test-mathematical

# With coverage
make test-coverage

# Performance benchmarks
make benchmark
```

## Documentation

### Code Documentation
- Document all public APIs with detailed docstrings
- Include mathematical examples in docstrings
- Use type hints consistently
- Document complex algorithms and mathematical concepts

### Mathematical Documentation
- Explain mathematical concepts for non-experts
- Provide references to relevant literature
- Include LaTeX source for mathematical expressions
- Document formalization strategies and design decisions

### User Documentation
- Maintain up-to-date installation instructions
- Provide comprehensive tutorials and examples
- Document configuration options and best practices
- Include troubleshooting guides

## Submission Guidelines

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] Mathematical correctness is verified
- [ ] Performance impact is acceptable
- [ ] Breaking changes are documented
- [ ] Examples are provided for new features

### Review Process

1. **Automated checks** - CI/CD pipeline runs tests and linting
2. **Code review** - Maintainers review code quality and design
3. **Mathematical review** - Domain experts verify mathematical correctness
4. **Integration testing** - Full workflow testing with multiple proof assistants
5. **Documentation review** - Ensure documentation is complete and accurate

### Merging Requirements

- All automated checks must pass
- At least one approving review from a maintainer
- Mathematical correctness verification for formalization changes
- No unresolved conflicts or requested changes

## Getting Help

### Community Resources

- **GitHub Discussions** - General questions and discussions
- **Issues** - Bug reports and feature requests
- **Discord/Slack** - Real-time community chat
- **Weekly Office Hours** - Direct access to maintainers

### Mathematical Expertise

- **Mathematics Advisory Board** - Expert guidance on formalization
- **Proof Assistant Experts** - Technical guidance for specific systems
- **Domain Specialists** - Subject matter expertise for specific mathematical areas

### Contributing Guidelines

- Start with small contributions to understand the codebase
- Ask questions early and often
- Participate in code reviews to learn best practices
- Share your mathematical expertise and domain knowledge

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to the project maintainers.

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project. See [LICENSE](LICENSE) for details.

---

Thank you for contributing to the advancement of automated mathematical formalization!