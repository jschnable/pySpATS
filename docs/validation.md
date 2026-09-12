# Validation of the rewrite

Validated locally on 2026-09-12 against the **supplied R SpATS 1.0-20 source**.
The R package and its dependencies were installed, and R itself generated the
reference files. Python fitting does not invoke R.

## Numerical agreement

Seventeen cross-language scenarios cover fixed and random genotypes;
PSANOVA, SAP and SAP.ANOVA; unequal degrees and penalty orders; nested bases;
centering; precision weights and offsets; missing responses and zero weights;
population-specific genotype variance; Poisson and binomial working models;
the real wheat trial with row and column random factors; and the existing
sorghum example.

| Cases | Largest observed absolute fitted-value discrepancy |
|---|---:|
| Synthetic PSANOVA, fixed/random genotypes | 1.4e-11 |
| Synthetic SAP, fixed/random genotypes | 2.8e-7 |
| Synthetic SAP.ANOVA | 3.4e-12 |
| Weighted/offset PSANOVA | 3.6e-8 |
| Missing-response/zero-weight PSANOVA | 1.7e-10 |
| Centered/population PSANOVA | 1.3e-11 |
| Poisson | 9.0e-14 |
| Binomial | 4.7e-12 |
| Unequal spline degrees | 3.4e-11 |
| Unequal SAP penalty orders | 2.2e-7 |
| Real wheat, fixed/random genotypes | 4.7e-5 |
| Existing sorghum trial, aligned input policy | 1.8e-5 |

All fourteen synthetic cases also compare R variance estimates, effective
dimensions, dispersion, restricted objective and **prediction standard errors**.
Their largest absolute SE discrepancy is below 1.8e-8. Random-genotype
heritability is compared without R's display rounding. Wheat tests compare
fitted values, variances and EDs. Sorghum checks fitted values and observation
count with its documented missing-data policy.

The tests use small explicit tolerances, not a universal bitwise-equality
claim. SVD signs may differ without affecting the model. Floating-point
objective changes can cross the convergence threshold on adjacent iterations,
especially for very small variance components. Variance parameters near a
boundary can have appreciable relative differences with negligible effect on
predictions; the sorghum interaction has ED about 0.00007. Differences at these
boundaries must not be interpreted as evidence for a meaningful spatial effect.

The sorghum CSV contains three blank genotype IDs, one with an observed yield.
R's default `read.csv` treats the blank as a genotype level; pandas treats it
as missing. The reference script explicitly treats blanks as missing and
removes incomplete predictor rows, matching the Python policy: 1,162 fitted
observations. Levels with no observed responses have NaN Python predictions;
reference comparisons exclude those undefined predictions. This policy is
explicit rather than concealed by a loose tolerance.

## Independent mathematical and behavioral checks

The suite also checks the mixed-equation residual, the coefficient covariance
against a full dense solve, fixed effects against an independently constructed
observation-space GLS calculation, and prediction SEs against the dense
covariance expression. These checks reduce the risk of reproducing the same
coding mistake in both fitting and its test.

Other checks cover offset equivalence, response-unit scaling, rank/confounding
errors, genotype estimability with fixed populations, missingness with duplicate
DataFrame indices, unchanged inputs, persistent prediction after pickling,
new-data ordering, unseen levels/extrapolation, actual plotted spatial values,
visible nonconvergence with synchronized returned parameters, exact variogram
pair calculations, axial direction/order invariance, and sequential/two-process
batch identity and error reporting.

Final development suite: **102 passed, no skipped tests**. The old tests tied
to retired experimental engines were removed rather than counted as evidence
for the new estimator. The R fixtures are checked into the repository so CI
requires no R installation. Regenerate with `Rscript tests/reference/generate.R`.

## Runtime and packaging

The measured benchmark, environment, caveats and R comparison are documented
in [benchmarks/README.md](../benchmarks/README.md). A 10,000-plot,
2,000-genotype Gaussian case took 0.33–0.46 seconds with nested bases and one
BLAS thread. A matched 2,000-plot case took about 0.014 seconds in Python and
0.14 seconds in R; both used 28 SAP iterations.

For the 0.3.0 release, a matched three-run comparison on the same 10,000-plot,
2,000-genotype CSV measured medians of 0.390 s Python versus 1.385 s R for
fixed genotypes (3.55×), and 0.576 s versus 6.091 s for random genotypes
(10.57×). Both used 71 and 121 iterations respectively. Maximum absolute
fitted-value differences were 1.12e-10 and 6.40e-11. Raw timings and settings
are saved under benchmarks/results/; these measurements supersede the earlier
single-run timings for direct R comparisons.

The actual wheat example and bundled sorghum example ran successfully. Wheel
and source distributions were built locally. Numerical tests ran on Python
3.14.7, NumPy 2.5.1, SciPy 1.18.0 and pandas 3.0.5. The advertised Python >=3.10
range is not a claim that every supported dependency/version combination has
been tested. The local Python 3.12 installation could not bootstrap a virtual
environment, so no 3.12 validation is claimed. The local shared 3.14 environment
also required `--no-build-isolation` for packaging because isolated build
subprocesses failed to import standard-library modules; the non-isolated build
succeeded.

## What this does not establish

Agreement with R validates this conversion, not every statistical assumption.
It does not demonstrate calibrated inference for separated/rare binary data,
outlier robustness, optimal knot selection, residual independence, genomic
relationship models, or multi-environment analysis. Covariance/SEs condition on
estimated variance parameters. The present fitter retains dense design and
coefficient-covariance storage; it is not an out-of-core solver. See the
[methodology review](methodology-review.md) for extensions requiring approval.
