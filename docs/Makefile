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

serve: autobuild notifywait

autobuild:
	sphinx-autobuild "$(SOURCEDIR)" $(BUILDDIR)/html -b html $(SPHINXOPTS) $(O) -p 4567 -H 0.0.0.0

notifywait:
	inotifywait -e close_write,moved_to,create -mr . --format '%f' | while read FILE; do echo $${FILE}; done
