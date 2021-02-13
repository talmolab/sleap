# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = ./build

# Export the BUILDDIR so we can pick it up in conf.py. We need this to
# be able to copy some of the files in _static to an alternative location
# of docs/_static in order to have paths work for README.rst and index.rst
# (which includes readme.rst)
export BUILDDIR

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

api.rst:
	python make_api_doctree.py

clean:
	rm -f api.rst
	rm -rf api
	rm -rf _autosummary

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile api.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)