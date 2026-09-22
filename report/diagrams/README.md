# Diagrams

Every figure in the report that is not a plot lives here as its own
standalone LaTeX file and is compiled to a PDF. `report-latex/*.tex` then
pulls the PDF in with `\includegraphics`, so the report itself contains no
TikZ any more and compiles much faster.

## Compiling

All diagrams at once, from this directory:

```sh
make -j8
```

Without `make`:

```sh
for f in diagram-*.tex; do pdflatex -interaction=nonstopmode -halt-on-error "$f"; done
```

A single diagram:

```sh
make diagram-probe.pdf          # or: pdflatex diagram-probe.tex
```

`make clean` removes the `.aux`/`.log` clutter but keeps the PDFs;
`make distclean` removes the PDFs too. Only `pdflatex` with a normal TeX Live
TikZ installation is needed, there are no external dependencies.

After changing a diagram, rebuild it here and then rebuild the report in
`../report-latex/`. The report does not rebuild diagrams for you.

## Files

| File | Figure in the report |
| --- | --- |
| `diagram-orbis.tex` | `fig:orbis`, the two-stage Orbis architecture |
| `diagram-probe.tex` | `fig:probe`, the attentive diffusion probe |
| `diagram-taxonomy.tex` | `fig:taxonomy`, generative / architecture / world-model taxonomy |
| `diagram-dit.tex` | `fig:dit`, the DiT block with adaLN-Zero |
| `diagram-planner-taxonomy.tex` | `fig:planner-taxonomy`, planner families |
| `diagram-navsim.tex` | `fig:navsim`, NAVSIM two-stage pseudo-simulation |
| `diagram-syn-pipeline.tex` | `fig:syn-pipeline`, synthetic data generation |
| `diagram-syn-arch.tex` | `fig:syn-arch`, the synthetic flow-matching denoiser |
| `diagram-navsim-pipeline.tex` | `fig:navsim-pipeline`, the NAVSIM probing pipeline |
| `diagram-navsim-mask.tex` | `fig:navsim-mask`, a padded batch and its frame mask |
| `diagram-aggregator-old.tex` | `fig:navsim-aggregator-old`, one pooled token per frame |
| `diagram-aggregator-new.tex` | `fig:navsim-aggregator-new`, N tokens as attention keys |
| `diagram-navsim-denoiser.tex` | `fig:navsim-denoiser`, the full denoiser on real data |

`diagramstyle.tex` holds the shared colour palette and node styles and is
`\input` by every diagram except `diagram-orbis.tex`, which still carries its
own inline copy of the same definitions. If you change the palette, change it
in both places, or switch `diagram-orbis.tex` over to `\input{diagramstyle}`.

## Conventions

- **Standalone, not part of the report.** Each file is a
  `\documentclass[tikz,border=8pt]{standalone}` document. That means `\cite`
  and `\ref` do not work inside a diagram, because there is no bibliography or
  label table to resolve against. Citations belong in the figure caption in
  the report, not in the drawing.
- **Relative placement.** Nodes are placed with the `positioning` library
  (`right=12mm of foo`, `below=9mm of bar`) and aligned with the `(a |- b)` /
  `(a -| b)` idiom. Where an offset genuinely cannot be expressed that way it
  is written once as a named `\coordinate` (`qLane`, `seqBack`, `condLane`)
  and everything else is anchored off that name, so no raw numeric pair is
  ever repeated.
- **Grouping.** Panels are drawn with `fit=` over an invisible node on a
  background layer, so a panel can never be the wrong size for its contents.
  `\panelbadge` puts a numbered badge and a title on a panel; `\legendchip`
  chains one legend swatch after another.
- **Padding.** The shared `dnode` style sets `inner xsep=5pt, inner ysep=4pt`
  and `minimum height=9mm`, so text never touches a border. Keep new nodes on
  one of the derived styles (`io`, `proc`, `latentn`, `learn`, `attn`, `head`,
  `cond`, `seqn`, `frozen`) rather than styling them ad hoc.
