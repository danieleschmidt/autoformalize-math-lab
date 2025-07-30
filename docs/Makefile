# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Custom targets for autoformalize-math-lab

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api/
	@echo "Build directory cleaned."

# Generate API documentation
apidoc:
	sphinx-apidoc -o api/ ../src/autoformalize --force --separate --module-first
	@echo "API documentation generated."

# Build HTML documentation with API docs
html-full: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation with live reload for development
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) \
		--host 0.0.0.0 --port 8000 --open-browser

# Check documentation for issues
check:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation in multiple formats
all: clean apidoc html latexpdf epub

# Build and serve documentation locally
serve: html
	cd $(BUILDDIR)/html && python -m http.server 8000

# Generate PDF documentation
pdf: latex
	cd $(BUILDDIR)/latex && make

# Build documentation for deployment
deploy: clean apidoc
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D html_baseurl="/autoformalize-math-lab/"
	@echo "Documentation built for deployment."

# Validate notebook execution
test-notebooks:
	pytest --nbval docs/tutorials/ docs/examples/

# Generate coverage report for documentation
doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)