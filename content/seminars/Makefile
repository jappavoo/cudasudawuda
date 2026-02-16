PAPER = papers
TEX = $(wildcard *.tex)

.PHONY: all clean

all: $(PAPER).pdf

$(PAPER).pdf: $(TEX)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.toc *.log *.out $(PAPER).pdf
