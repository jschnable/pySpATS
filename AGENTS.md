# Working on pySpATS

Read README.md and docs/methodology-review.md before changing model behavior.
The statistical reference is the supplied R SpATS 1.0-20 archive in rSpATS/.
Do not silently replace its model/estimation assumptions: propose statistical
extensions to the user before implementing them. Algebraically equivalent
numerical improvements and corrections to the Python conversion are in scope.

Production path: core.py -> model_basis.py + engine.py. There is one fitting
engine. batch.py orchestrates independent fits; it is not a multi-environment
model. Avoid reintroducing experimental alternate engines.

Key invariants:
- Five PSANOVA components, full tensor null space, R basis/penalty scaling.
- SAP overlapping penalties are not equivalent to five independent variances.
- Precision is penalty/variance; residual precision is weights/dispersion.
- Exact effective dimensions; never substitute parameter counts.
- Heritability requires random genotypes and the estimable genotype dimension.
- Adjusted plot phenotypes and genotype effects are different outputs.
- Convergence status must describe the returned coefficient/variance iterate.
- Preserve input row order/index; do not mutate input data or global RNG state.

Run python -m pytest -q. R is optional for ordinary tests. Never regenerate R
reference outputs using Python. tests/reference/generate.R is the independent
oracle. Changes to tolerances need a documented numerical reason.

README documents current APIs. Old notebooks and saved example CSV/PNG outputs
are historical artifacts, not current numerical evidence. Prefer the current
wheat, sorghum and batch scripts. Optional plotting must not become an import-time
runtime dependency. Keep package metadata in pyproject.toml.
