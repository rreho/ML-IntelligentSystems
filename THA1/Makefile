# make        -> run the code (tables, figures) and build report.pdf
PY = python3

all: code report

code:
	$(PY) code/data.py
	$(PY) code/transform.py
	$(PY) code/least_squares.py
	$(PY) code/gd_linear.py

report:
	latexmk -pdf -interaction=nonstopmode report.tex

clean:
	latexmk -C report.tex
	rm -rf tables figs

.PHONY: all code report clean
