# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = scanner
SOURCEDIR     = .
BUILDDIR      = ../build/docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

watch: autobuild notifywait

autobuild:
	sphinx-autobuild -b html "$(SOURCEDIR)" $(SPHINXOPTS) $(O) $(BUILDDIR)/html -p 6006 -H 0.0.0.0

notifywait:
	inotifywait -mr .. @../build @../thirdparty @../dist @../docs --format '%f' | while read i; do cd .. && python3 python/setup.py bdist_wheel && cd docs && touch Makefile; done
