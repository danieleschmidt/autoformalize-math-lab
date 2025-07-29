# Makefile for autoformalize-math-lab
# A comprehensive build system for mathematical formalization project

.PHONY: help install install-dev install-provers clean test test-unit test-integration test-mathematical lint format type-check security-check docs docs-serve benchmark profile docker docker-run release pre-commit

# Default target
help: ## Show this help message
	@echo "autoformalize-math-lab - Mathematical Formalization Workbench"
	@echo "============================================================"
	@echo
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Python and environment settings
PYTHON := python3
PIP := pip
VENV_DIR := venv
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Installation targets
install: ## Install the package for production use
	$(PIP) install -e .

install-dev: ## Install the package with development dependencies
	$(PIP) install -e ".[dev,docs]"
	$(MAKE) install-pre-commit

install-pre-commit: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

install-provers: ## Install proof assistants (Lean 4, Isabelle, Coq)
	@echo "Installing proof assistants..."
	@echo "Installing Lean 4..."
	curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
	@echo "Lean 4 installation completed. Please restart your shell or run 'source ~/.profile'"
	@echo
	@echo "For Isabelle: Download from https://isabelle.in.tum.de/"
	@echo "For Coq: Install via package manager or opam"
	@echo "  - Ubuntu/Debian: sudo apt-get install coq"
	@echo "  - macOS: brew install coq"
	@echo "  - Via opam: opam install coq"

setup-env: ## Set up development environment from scratch
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Created virtual environment. Activate with:"
	@echo "  source $(VENV_DIR)/bin/activate"

# Cleaning targets
clean: ## Clean build artifacts and temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	find . -type f -name "*.rej" -delete

clean-all: clean ## Clean everything including virtual environment
	rm -rf $(VENV_DIR)/
	rm -rf .venv/
	rm -rf outputs/
	rm -rf temp/
	rm -rf formalization_cache/
	rm -rf verification_logs/
	rm -rf arxiv_cache/
	rm -rf papers_cache/

# Testing targets
test: ## Run all tests
	pytest $(TEST_DIR) -v

test-unit: ## Run unit tests only
	pytest $(TEST_DIR)/unit -v

test-integration: ## Run integration tests
	pytest $(TEST_DIR)/integration -v

test-mathematical: ## Run mathematical correctness tests
	pytest $(TEST_DIR)/mathematical -v -m mathematical

test-coverage: ## Run tests with coverage report
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-lean: ## Run Lean 4 specific tests
	pytest $(TEST_DIR) -v -m lean

test-isabelle: ## Run Isabelle/HOL specific tests
	pytest $(TEST_DIR) -v -m isabelle

test-coq: ## Run Coq specific tests
	pytest $(TEST_DIR) -v -m coq

test-llm: ## Run tests requiring LLM API access (requires API keys)
	pytest $(TEST_DIR) -v -m llm

test-fast: ## Run fast tests only (exclude slow and expensive tests)
	pytest $(TEST_DIR) -v -m "not slow and not expensive"

# Code quality targets
lint: ## Run linting with flake8
	flake8 $(SRC_DIR) $(TEST_DIR)

lint-ruff: ## Run linting with ruff (faster alternative)
	ruff check $(SRC_DIR) $(TEST_DIR)

format: ## Format code with black and isort
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

format-check: ## Check if code is properly formatted
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR)

security-check: ## Run security checks with bandit
	bandit -r $(SRC_DIR) -f json -o security-report.json || true
	bandit -r $(SRC_DIR)

qa: lint type-check security-check ## Run all quality assurance checks

# Documentation targets
docs: ## Build documentation
	cd $(DOCS_DIR) && make html

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	cd $(DOCS_DIR) && make clean

docs-api: ## Generate API documentation
	sphinx-apidoc -o $(DOCS_DIR)/api $(SRC_DIR) --force

# Performance and profiling targets
benchmark: ## Run performance benchmarks
	pytest $(TEST_DIR)/performance -v --benchmark-only

profile: ## Run profiling on sample formalization
	$(PYTHON) -m cProfile -o profile.stats scripts/profile_formalization.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

performance-test: ## Run performance regression tests
	pytest $(TEST_DIR)/performance -v

# Docker targets
docker: ## Build Docker image
	docker build -t autoformalize-math-lab:latest .

docker-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t autoformalize-math-lab:dev .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace autoformalize-math-lab:latest

docker-test: ## Run tests in Docker container
	docker run --rm -v $(PWD):/workspace autoformalize-math-lab:latest make test

# Dataset and evaluation targets
download-datasets: ## Download benchmark datasets
	$(PYTHON) scripts/download_datasets.py

evaluate: ## Run full evaluation on benchmark datasets
	$(PYTHON) scripts/run_evaluation.py --output evaluation_results/

evaluate-fast: ## Run quick evaluation on sample dataset
	$(PYTHON) scripts/run_evaluation.py --sample 100 --output evaluation_results/sample/

# Development workflow targets
dev-setup: setup-env install-dev install-provers ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "Don't forget to:"
	@echo "1. Activate your virtual environment: source $(VENV_DIR)/bin/activate"
	@echo "2. Set up your .env file with API keys"
	@echo "3. Run 'make test-fast' to verify installation"

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

ci-check: format-check lint type-check test-fast security-check ## Run CI/CD checks locally

# Release and packaging targets
build: clean ## Build distribution packages
	$(PYTHON) -m build

release-test: build ## Upload to test PyPI
	twine upload --repository testpypi dist/*

release: build ## Upload to PyPI
	twine upload dist/*

version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

# Utility targets
check-deps: ## Check for dependency vulnerabilities
	safety check

update-deps: ## Update dependencies to latest versions
	pip-tools compile --upgrade requirements.in
	pip-tools compile --upgrade requirements-dev.in

freeze-deps: ## Freeze current dependencies
	pip freeze > requirements-frozen.txt

# Mathematical formalization specific targets
validate-proofs: ## Validate all generated proofs
	$(PYTHON) scripts/validate_all_proofs.py

generate-examples: ## Generate example formalizations
	$(PYTHON) scripts/generate_examples.py --output examples/

run-formalization: ## Run interactive formalization session
	$(PYTHON) -m autoformalize.cli

formalize-file: ## Formalize a specific LaTeX file (usage: make formalize-file FILE=path/to/file.tex)
	$(PYTHON) -m autoformalize.cli formalize $(FILE) --output $(FILE:.tex=.lean)

# Monitoring and logging targets
view-logs: ## View recent formalization logs
	tail -f logs/autoformalize.log

clean-logs: ## Clean old log files
	find logs/ -name "*.log" -mtime +7 -delete

# Database and cache management
init-db: ## Initialize database
	$(PYTHON) scripts/init_database.py

clean-cache: ## Clean formalization cache
	rm -rf formalization_cache/*
	rm -rf verification_logs/*

backup-cache: ## Backup formalization cache
	tar -czf cache_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz formalization_cache/ verification_logs/

# Special targets for mathematical domains
test-algebra: ## Test algebraic formalization
	pytest $(TEST_DIR) -v -k "algebra"

test-analysis: ## Test real analysis formalization
	pytest $(TEST_DIR) -v -k "analysis"

test-topology: ## Test topology formalization
	pytest $(TEST_DIR) -v -k "topology"

test-number-theory: ## Test number theory formalization
	pytest $(TEST_DIR) -v -k "number_theory"

# Debug and development helpers
debug: ## Run formalization in debug mode
	$(PYTHON) -m autoformalize.cli --debug

shell: ## Start interactive Python shell with autoformalize loaded
	$(PYTHON) -c "import autoformalize; print('autoformalize loaded'); import IPython; IPython.start_ipython()"

notebook: ## Start Jupyter notebook for development
	jupyter notebook examples/

# Information targets
info: ## Show project information
	@echo "autoformalize-math-lab - Mathematical Formalization Workbench"
	@echo "============================================================"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Project directory: $(PWD)"
	@echo "Source directory: $(SRC_DIR)"
	@echo "Test directory: $(TEST_DIR)"
	@echo
	@echo "Proof assistants:"
	@echo "  Lean 4: $(shell lean --version 2>/dev/null || echo 'Not installed')"
	@echo "  Isabelle: $(shell isabelle version 2>/dev/null || echo 'Not installed')"
	@echo "  Coq: $(shell coqc --version 2>/dev/null || echo 'Not installed')"

env-check: ## Check environment setup
	@echo "Checking environment setup..."
	@$(PYTHON) -c "import sys; print(f'Python: {sys.version}')"
	@$(PYTHON) -c "import autoformalize; print('autoformalize package: OK')" 2>/dev/null || echo "autoformalize package: NOT INSTALLED"
	@echo "Environment variables:"
	@echo "  OPENAI_API_KEY: $(if $(OPENAI_API_KEY),SET,NOT SET)"
	@echo "  ANTHROPIC_API_KEY: $(if $(ANTHROPIC_API_KEY),SET,NOT SET)"

# Default target when no target is specified
.DEFAULT_GOAL := help