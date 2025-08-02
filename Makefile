# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/out

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@sphinx-apidoc -f -o "$(SOURCEDIR)" "pycheribenchplot"

tools: dwarf_scraper

dwarf_scraper: tools/dwarf-scraper/CMakeLists.txt
	mkdir -p build
	cmake -B build/dwarf-scraper -S tools/dwarf-scraper -GNinja -DCHERISDK=${CHERISDK}
	ninja -C build/dwarf-scraper

.PHONY: apidoc help tools dwarf_scraper Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile apidoc
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
