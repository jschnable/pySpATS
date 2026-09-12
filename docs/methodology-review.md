# Review of the R methodology and the Python port

Baseline: the supplied `rSpATS/SpATS_1.0-20.tar.gz`, package date 2026-05-17.
This is the supplied reference release, not an inferred reconstruction of the
R release used in 2025. The rewrite retains its spatial mixed-model approach.
The changes in statistical assumptions proposed below have **not** been implemented.

## What should be retained

The spatial model fits genotype, experimental covariates, independent random
factors and the spatial surface jointly. Correcting the phenotype first and
then estimating genotype effects can change the answer and understate uncertainty.
The five-component PS-ANOVA decomposition separates two marginal smooths, two
smooth-by-linear terms, and a smooth-by-smooth interaction. Its unpenalized
null space also contains the row-by-column product. `SAP` and `SAP.ANOVA` share
some variance parameters across blocks and are distinct models, not synonyms.

Second-difference penalties act on spline coefficients, not on measured yields.
The interaction penalty in PSANOVA is a **sum** of marginal penalties. The
SVD rotations, marginal constant/linear scaling, and nested bases matter for
variance estimates. Rotating signs within an equivalent basis does not change
fitted values. Projecting smooths against arbitrary covariates changes their
covariance model and is not a harmless cleanup step.

These choices follow `MM.basis.R`, `construct.2d.pspline.R`, `SpATS.R` and
`compute.hat.diagonal.R`. They are consistent with the authors' description of
joint tensor P-spline spatial correction and nested interaction bases:
[Rodríguez-Álvarez et al. (2018)](https://doi.org/10.1016/j.spasta.2017.10.003).

## Defects in the old Python port, corrected

| Finding | Consequence | Replacement |
|---|---|---|
| `core.py` hard-coded three spatial blocks and a three-column polynomial | Different model from R PSANOVA; missing smooth-by-linear terms and the polynomial interaction | Complete R basis and configurable PSANOVA/SAP/SAP.ANOVA |
| `solver.py` multiplied penalties by numbers updated as variances | Reversed variance/precision roles and incorrect residual scaling | Precision is the sum of penalty diagonals divided by their variances; observation precision is W/psi |
| Effective dimensions used half the block size or nominal counts | Incorrect smoothing updates, residual variance and diagnostics | Exact covariance traces at every SAP iteration |
| Fixed-genotype heritability and ED/count shortcuts | A number called heritability without the required genetic variance model | Random genotypes only; denominator rank([X,Zg])-rank(X), including other fixed effects |
| Offset was not subtracted from the working response | Biased coefficients when offsets were nonzero | Fit y-offset for Gaussian, and the equivalent working response for GLMMs |
| Missing responses flowed into arithmetic as NaN despite zero weights | Zero times NaN contaminated fits | Select observed, positive-weight rows before linear algebra |
| One Gaussian outer iteration obscured inner convergence; nominal diagnostics | Results could appear successful without estimated smoothing convergence | Explicit SAP/outer convergence flags and history; warnings on exhaustion |
| `load_wheatdata()` generated synthetic data | Validation appeared to use an established field trial but did not | Bundle the actual 330-plot, 107-variety R dataset with attribution |
| Plotting reconstructed effects through obsolete internals | Spatial plots could misrepresent the fitted model | Plot the fitted spatial contribution directly |
| Directional variogram selected a directed angle from unordered pairs | Reordering plots could change the diagnostic | Axial angles modulo 180, each pair once, exact streamed bins |

The disconnected CHOLMOD/experimental engines and tests of their internal
three-block assumptions were retired. Their history is in git. Tests now
exercise one production engine and include frozen R outputs, independent
marginal-GLS calculations, full mixed-equation/covariance checks, prediction,
missingness, offsets, confounding, batch jobs and real wheat data.

## Improvements to R's implementation, preserving its model

- **Avoid quadratic observation-space allocation in basis construction.**
  `MM.basis.R`, decomposition 4, creates an outer product of an all-ones vector
  just to center a small null-space basis. Column means do the same computation
  in linear memory. Python uses the latter.
- **Validate fixed rank once per working system.** `SpATS.R` calls `qr(K)`
  inside every variance iteration. The unpenalized design determines exact
  fixed confounding; genotype elimination permits a small rank check. Python
  fails with an actionable error rather than adding a ridge to fixed effects.
- **Retain genotype elimination.** R already uses this good idea in
  `construct.henderson.matrix.R`; it is not a new statistical method invented
  by this rewrite. Python accumulates genotype cross-products by group and
  factors the remaining Schur system. Full coefficient covariance is assembled
  once after the final iteration.
- **Scale the linear system and stabilize traces.** Diagonal equilibration is
  algebraically equivalent. Evaluating data contributions directly avoids loss
  of significant digits in `1 - precision * posterior_variance` near zero
  component variance. A numerical variance floor of 1e-50 guards division;
  it is not evidence that boundary components are positive in the population.
- **Expose convergence honestly.** The R stopping rule uses an absolute change
  in the restricted objective; its default is 1e-3. Python defaults to 1e-6 and
  records the objective, variances and EDs, and warns when an iteration limit
  is reached. It retains the same stopping criterion for reference comparability.
- **Keep estimates synchronized.** The R loop can update variance parameters
  at exhaustion after computing coefficients from the preceding iterate.
  Python returns coefficients, covariance, EDs and variances from one iterate.
- **Validate input semantics.** Negative/nonfinite weights, infinite responses,
  ambiguous formula strings, duplicate terms and inconsistent genotype
  populations receive explicit errors. The source's predictor-NA screening
  does not enumerate every random-effect column; the Python mask does.
- **Do not round scientific outputs prematurely.** R `getHeritability.R`
  rounds to two decimals; Python returns full precision.

## Statistical extensions that require your approval

These are candidates for experiments, not claims that they are universally
better than the supplied R model. The first two are the most useful next studies.

| Priority | Candidate | Why investigate | What changes; validation before adoption |
|---|---|---|---|
| 1 | Residual covariance, e.g. separable AR(1) by row/column plus a nugget | Smooth trends and independent row/column factors may leave short-range residual correlation | Replaces diagonal residual covariance. Compare calibration, held-out genotype prediction and residual diagnostics on multiple real trials. Check confounding between smooth and residual spatial components. Do not select on apparent heritability alone. |
| 2 | Sensitivity analysis across spline resolutions, with spatially blocked validation | R's fixed default grid may underfit some fields or be needlessly expensive for others | Automating selection changes the fitting procedure even if each candidate is an R model. Compare prediction and stability, specify blocks without phenotype leakage, and report selected settings. Existing explicit nseg/nest_div controls are already implemented. |
| 3 | Robust residual models for damaged/mislabeled plots | Squared-error fitting can let a few bad measurements distort a surface | A heavy-tailed likelihood or robust loss changes estimation and inference. Evaluate against genuine extreme phenotypes; do not silently delete biological extremes. |
| 4 | Relationship-matrix and multi-environment genetic effects | Independent genotype BLUPs cannot represent kinship or genotype-by-environment covariance | Adds covariance structure and changes interpretation of variance and heritability. Requires aligned IDs, PSD checks, connectedness diagnostics, and independent reference fits. The batch API currently fits separate fields. |
| 5 | Smoothing/variance-parameter uncertainty | Conditional SEs omit uncertainty in estimated variance parameters | Parametric bootstrap or other calibrated inference adds computation and assumptions. Compare interval coverage under realistic replication/missingness. Current SEs are explicitly conditional. |
| 6 | Stronger variance-boundary/convergence diagnostics or a second optimizer | An absolute objective plateau need not imply well-determined variance components | Score checks or optimization of the same REML objective can improve diagnosis; a replacement optimizer needs benchmarking and boundary agreement before becoming default. No replacement optimizer was silently introduced. |
| 7 | More accurate generalized mixed-model inference | R's Poisson/binomial working mixed-model method is approximate | Laplace integration or Bayesian inference changes the estimation approach. Compare on sparse counts, rare binary outcomes and separation; matching R does not establish inferential calibration there. |

R's generalized heritability is trial-specific, not a direct estimate of a
universal narrow-sense heritability parameter. The nominal-dimension correction
is intentional. Neither high heritability nor a very smooth residual map alone
proves that genetic signal has been preserved. The sorghum study provides a
useful model for evaluating real trials against alternative spatial analyses:
[Velazco et al. (2017)](https://doi.org/10.1007/s00122-017-2894-4).

## Current scope and remaining limitations

This is a tested reimplementation of the fitting mathematics and a deliberately
explicit Python API, not a parser for the entire R formula/prediction language.
Fixed/random inputs are additive column lists; construct interaction columns
explicitly. R's `predict(which=..., predFixed='marginal')` is not replicated by
`genotype_predictions()`, which clearly reports effects. Plot predictions and
conditional SEs support complete new rows inside the fitted field; unseen
factor levels and extrapolation are rejected. Levels with no fitted responses
produce NaN plot predictions with a warning.

The fitter still stores dense design arrays and full coefficient covariance.
The genotype Schur solve reduces factorization cost but does not eliminate
O(n*k + k²) storage. Large spline interaction bases can be expensive even with
few genotypes. There is no automatic downsampling, approximate ED, or hidden
change of segment count. Gaussian raw response residuals are exposed; GLMM
response residuals are not labeled R deviance residuals. Spatial-only adjusted
phenotypes are provided for Gaussian models only.
