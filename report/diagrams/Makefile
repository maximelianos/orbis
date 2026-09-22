# Build every standalone diagram in this folder into a PDF.
#
#   make            build all diagrams that are out of date
#   make -j8        the same, in parallel
#   make clean      remove LaTeX aux files, keep the PDFs
#   make distclean  also remove the generated PDFs
#   make diagram-probe.pdf     build one diagram

TEXFILES := $(wildcard diagram-*.tex)
PDFFILES := $(TEXFILES:.tex=.pdf)

LATEX      ?= pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error

all: $(PDFFILES)

# every diagram also depends on the shared preamble
%.pdf: %.tex diagramstyle.tex
	$(LATEX) $(LATEXFLAGS) $<

clean:
	rm -f *.aux *.log *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDFFILES)

.PHONY: all clean distclean
