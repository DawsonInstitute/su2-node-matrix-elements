# SU(2) Node Matrix Elements

This repository contains the paper "Closed-Form Matrix Elements for Arbitrary-Valence SU(2) Nodes via Generating Functionals" and its implementation as a GitHub Pages site.

**Website**: [https://dawsoninstitute.github.io/su2-node-matrix-elements/](https://dawsoninstitute.github.io/su2-node-matrix-elements/)

## Abstract

We derive closed-form expressions for SU(2) operator matrix elements on arbitrary-valence nodes by extending the universal generating functional approach with source terms. Our central result is a determinant-based formula incorporating group-element dependence, which yields all matrix elements via a single Gaussian integral and hypergeometric expansion.

## Repository Contents

- `index.html`: HTML version of the full paper with MathJax support
- `_layouts/default.html`: Jekyll layout template
- `_includes/head.html`: HTML head section with CSS and script references
- `assets/css/style.scss`: Main CSS for the site (processed by Jekyll)
- `_config.yml`: Jekyll configuration file
- Original LaTeX source files


## Scope, Validation & Limitations

- Scope: The materials and numeric outputs in this repository are research-stage examples and depend on implementation choices, parameter settings, and numerical tolerances.
- Validation: Reproducibility artifacts (scripts, raw outputs, seeds, and environment details) are provided in `docs/` or `examples/` where available; reproduce analyses with parameter sweeps and independent environments to assess robustness.
- Limitations: Results are sensitive to modeling choices and discretization. Independent verification, sensitivity analyses, and peer review are recommended before using these results for engineering or policy decisions.
