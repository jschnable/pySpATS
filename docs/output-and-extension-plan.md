Current update: the user approved balanced references with overrides.
`adjusted_plots()` and `genotype_means()` are now implemented alongside
`variance_partition()`. See README for the definitive current API. Spatial
averaging and numeric means count observed positive-weight plots equally;
known offsets default to their mean with an explicit override. The discussion
below records the design history, including earlier pending decisions.

# Proposed outputs and extension sequence

Status: the current agreement below supersedes the original proposal retained
later in this document. The estimator remains the R-validated baseline.

## Current scope: three standard tables

1. **Adjusted plots**: one row per input plot, retaining its observed phenotype,
   genotype, residual, and identifiers. The adjusted phenotype standardizes all
   nuisance effects to shared reference conditions while retaining genotype and
   plot residual. Missing observations stay missing in the adjusted column;
   predictions belong in a separate column.
2. **Genotype values**: one row per genotype, environment and trait, containing
   its predicted phenotype at the same reference as the adjusted plots, with
   uncertainty and observed plot count. No plot residual is included. Fixed
   genotypes yield adjusted BLUEs; random genotypes yield shrunken predictions.
3. **Variance partition**: model-assigned random-effect and residual variances
   and percentages, including either combined or detailed spatial components.
   Fixed terms are explicitly listed without variance components.

All three tables will carry environment and trait identifiers. Reference
conditions must be stored alongside the first two outputs so results can be
interpreted and reproduced. A future joint model will supply environment-specific
rows with the same identity columns. Overall genotype values across environments
will require explicit environment weights.

**Earlier implementation status (superseded):** `variance_partition()` and single-field
`environment_id` metadata are implemented in the working tree. Adjusted plot
and genotype output APIs await discussion and approval of reference definitions.
Existing `to_frame()` and `genotype_predictions()` retain their documented
behavior and do not yet implement items 1 and 2 above.

### Proposed shared reference, awaiting agreement

For Gaussian traits, adjusted plot = observed phenotype minus fitted nuisance
contribution at that plot plus nuisance contribution at the reference. Genotype
value = fitted genotype prediction at that reference, without a plot residual.
Nuisance effects include spatial effects, non-genotype random factors, fixed
covariates, and any known offset. Genotype effects are retained.

Proposed defaults: numeric covariates at common means over fitting plots;
fixed categorical factors averaged equally over their fitted levels; nuisance
random factors at zero; spatial contribution averaged over the observed field
using a documented common weighting scheme. User overrides should allow a
specific treatment, numeric value or known offset. The reference is shared by
all genotypes, never taken from each genotype's own occupied plots. Spatial
averaging preserves the field's phenotype level while removing location effects.

These choices are still proposals. The user has been asked whether to default
to a balanced reference with overrides or require explicit reference conditions.
Offsets and spatial averaging weights must also be made explicit before the
adjustment APIs are implemented. Non-estimable references must be identified
rather than assigned misleading predictions or standard errors.

### Implemented variance definition

For each independent random coefficient block, report `trace(Z G Z') / n` over
observed positive-weight plots, with fitted prior covariance G. For ordinary
random intercepts this equals the estimated variance parameter. For genotype
populations, contributions include each population's fraction of observed plots.
Residual contribution is mean(`psi / weight`). Percentages use the sum of these
contributions. Fixed terms, including the spatial polynomial, are outside this
budget. SAP components use disjoint covariance blocks, not overlapping penalties.
The method does not change fitting and currently supports Gaussian models.

Check comparisons, diagnostics and downstream covariance exports remain useful
extensions, but are not additional standard tables.

## Earlier detailed proposal (superseded where it differs above)

## 1. Keep the everyday outputs to three tables

### A. Genotype performance

Proposed call: `result.genotype_means(...)`.

Question answered: What is each genotype's expected phenotype under the same
reference conditions, and how precisely is it estimated?

Default columns:

| Column | Meaning |
|---|---|
| genotype | Original genotype identifier; retain input type |
| estimate | Adjusted mean in the trait's units |
| se | Conditional standard error for the reported mean/prediction |
| n_obs | Number of observed, positive-weight plots informing that genotype |
| status | Estimated, model-dependent prediction, or unavailable, with reason |

An optional `by="Treatment"` adds treatment columns and one row per
 genotype/treatment combination. A reported observation count then describes
that cell, not all observations of the genotype. Return uncertainty even when
there is only one observed plot; do not imply equal precision across entries.

The table does not rank genotypes by default. An explicit direction is needed
before reporting top entries: higher yield, lower disease score, or a desired
target height are different selection objectives. Do not add a generic
"best genotype" flag.

A small result metadata record identifies the response, units if supplied,
fixed/random genotype treatment, prediction target, averaging rules, scope,
model version and convergence. Optional approximate conditional intervals are
labeled with their method; default output is estimate plus SE. Heritability
stays in the model summary and is unavailable for fixed genotypes.

Reference-condition rules must be part of the API contract:

- All genotypes are evaluated under the same conditions. Never use each
  genotype's own distribution of locations/treatments as its reference.
- Categorical fixed factors are equally averaged by default; `averaging="observed"`
  or explicit weights permit other scientifically intended targets. Precision
  weights used during fitting are not automatically target-population weights.
- Numeric covariates are evaluated at common observed means unless overridden
  with `at={...}`. These values are reported in metadata.
- Spatial contributions use a centered field reference: the common field
  level is retained while location-specific deviations are removed. Specify
  the common set of fitted plot locations and its averaging weights, and
  propagate the corresponding coefficient transformation into the SEs.
  Reporting must not depend on the original basis's arbitrary constant terms
  or the fit-time `center` flag. Validate against an equivalent centered R fit
  using the same reference. This is an explicit reporting estimand, not a
  change to the fitted model.
- Non-genetic block/row/column random effects are evaluated at their population
  mean of zero. Retain genotype BLUPs for a random-genotype model.
- Known offsets default to zero in the reference, with an explicit reference
  offset permitted and recorded. Do not accidentally inherit genotype-specific
  offsets from the original plot rows.
- Conditioning on treatment is not evidence that genotype-by-treatment
  interaction was fitted. With today's additive model, treatment changes are
  shared across genotypes. Report that limitation explicitly.
- When richer designs arrive, do not average over impossible nested-factor
  combinations or manufacture estimates for confounded cells. Require valid
  reference combinations and report estimability/support.

### B. Comparisons against checks or nominated entries

Proposed call: `result.compare_genotypes(reference="TX430", ...)`.

Question answered: How much better or worse is each genotype than a named
check under the same reference conditions?

Default columns: genotype, reference, optional grouping columns, difference,
se_difference, status. Optional conditional intervals and explicitly requested
percent differences may be added; percentages require a meaningful nonzero
reference. Differences are genotype minus reference, irrespective of whether
higher is desirable.

Support one check, a short check list, or an explicit list of pairs. Do not
materialize all pairwise differences by default: 2,000 genotypes imply nearly
two million unordered comparisons. Compute requested contrasts directly.

Use covariance when calculating SEs of differences. Adding two squared SEs
and assuming independence is wrong when the estimates share fitted effects.
No automatic significance stars, letter groups or unadjusted mass-testing
p-values. Simultaneous inference can be a separately specified later feature.

### C. Plot results

Keep `result.to_frame()` as the existing entry point.

Retain original identifiers/columns and expose fitted value, response residual,
centered spatial trend, spatial-only adjusted response and used_for_fit. Add
an explicit exclusion/unavailable-prediction reason. Preserve every input row
and its order, including rows with no usable response or genotype.

Do not label plot-level adjusted values as genotype means: they still contain
plot noise and may retain treatment/block contributions. For future robust
fits, distinguish supplied precision weights from estimated robustness weights
and record what changed. No automatic deletion of flagged observations.

Existing diagnostics (`summary`, `plot`, `variogram`) complement these three
tables; this proposal does not add another overlapping collection of summaries.

## 2. One downstream export, with uncertainty preserved

Proposed call: `result.export_stage1(directory, ...)`.

Write:

- `means.csv`: the genotype mean table, with stable row IDs and environment/trait
  keys when supplied.
- `covariance.npz`: covariance in exactly the same row order, plus stored row IDs.
- `metadata.json`: the explicit reference conditions, model/fitting settings,
  genotype treatment, uncertainty interpretation, exclusions and software version.

Make fixed-genotype estimates the initial supported input to this export.
A model fitted with random genotypes should receive an explanation that its
BLUP table is not an automatically interchangeable first-stage dataset. Offer
an explicit fixed-genotype refit through the existing fitter; never refit or
deregress silently. For genuinely unestimable fixed genotype comparisons,
report the limitation; no convenient numerical export can create connectedness.

Preserve covariance rather than just exporting 1/SE². A diagonal approximation
may be exported if requested, clearly marked as such. For random-genotype
predictions, distinguish prediction-error covariance from the covariance of
fixed-effect estimates; do not feed them into a second-stage model as though
they were interchangeable.

The rationale is established in [Endelman (2023), fully efficient two-stage
analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10033618/): first-stage
precision and covariance matter for subsequent analysis. This does not imply
that every two-stage specification exactly reproduces a joint fit.

## 3. Implementation and acceptance criteria for these outputs

1. Define a single internal reference/contrast builder. Store stable mappings
   from design columns to terms and levels instead of relying on display names.
   Represent each requested mean/difference by a linear combination L of the
   existing fixed and random coefficients.
2. Compute estimates and covariance consistently as L b and L C L'. Compute
   only diagonal uncertainties or selected contrasts when full covariance is
   not requested. Do not allocate an expanded genotype-by-every-plot table.
3. Add genotype means and explicit check comparisons. Keep the existing
   `genotype_predictions()` method as the documented effect table for compatibility.
4. Add exclusion reasons to plot output and the stable first-stage export.
   CSV metadata must survive export in JSON; DataFrame.attrs alone is insufficient.
5. Validate with independent full-matrix calculations, R marginal predictions
   under matched reference conventions, and reparameterization checks. Include
   unbalanced treatments, offsets, fixed/random genotypes, population groups,
   missing responses, an unavailable check, shuffled/non-string IDs, and covariance
   alignment. Equivalent basis centering/contrast coding must not alter estimates.
6. Keep all existing 102 checks passing and benchmark reporting on the
   10,000-plot/2,000-genotype case. Ordinary genotype tables should not need full
   all-pairs covariance construction.

Initial output implementation scope: Gaussian traits. Count/binary adjusted
means need an explicit choice between averaging on the link scale and averaging
response-scale predictions, and between conditioning and integrating random
effects. Preserve existing plot prediction for those models; do not quietly
apply Gaussian mean/SE formulas to nonlinear response-scale means.

## 4. Explore spline sensitivity (#3) and robustness (#5) together

### Spline sensitivity first

Start with a small explicit candidate set of nseg/nest_div values under the
same experimental design, family, genotype treatment and observation set.
Produce a compact candidate table: settings, convergence, runtime, spatial ED,
validation error where defined, genotype-mean changes, and requested selection
changes. Keep the original fit alongside alternatives.

Ask two distinct questions: is the predicted phenotype stable, and would the
breeder's selection decision change? High overall correlation can conceal
changes among the entries near a selection threshold. A selection-stability
metric requires an explicit direction and selected fraction or nominated set.

Use spatial holdouts appropriate to the intended prediction task. Folds must
retain genotype estimability when predicting known-genotype performance. In
partially replicated/augmented designs, some holdouts cannot provide that test;
record them, rather than dropping them silently or claiming that ordinary
random-fold prediction validates spatial correction. Do not select by the
largest heritability or smallest training residual error.

### Influence analysis, then an optional robust estimator

First report which plots have the largest residuals/influence and how targeted
refits affect genotype means and check comparisons. A large residual alone is
not proof of bad data. Initially refit selected influential plots, avoiding a
full leave-one-plot-out fit for every plot of every large field.

After that diagnostic is useful, experimentally compare a robust estimator
(e.g. a carefully specified Huber procedure or Student-t residual model) with
the Gaussian baseline. Choosing the residual model, variance-estimation method
and uncertainty calculation requires a statistical design decision; iterated
Gaussian weights alone should not be mislabeled an exact Student-t likelihood.

Use simulations with known genetic/spatial signals and real-data sensitivity:
clean Gaussian errors, isolated measurement mistakes, clusters of damaged plots,
true extreme genotypes and genotypes with little replication. Measure recovery
of genetic effects, selection errors, interval calibration and compute cost.

A small crossed comparison of spline resolution and robustness is useful:
flexible surfaces can absorb bad points, while an overly rigid surface can make
legitimate spatial variation look like outliers. Do not change both defaults
without evidence identifying which adjustment helps.

## 5. Pair richer designs (#8) with multi-environment modeling (#9)

They should share a design/model layer. A formula parser alone is insufficient.
The layer needs fixed/random interactions, crossed versus nested factors,
known genotype IDs across trials, and separate physical-field identities.
Examples include environment, genotype, genotype-by-environment,
genotype-by-treatment, replication within field, and block within replication.

Separate **physical field** from **evaluation environment**. Low- and
high-nitrogen treatments sharing a physical field can share its spatial surface
while having different genotype responses. Conversely, row 1/column 1 at two
locations never denotes adjacent plots or a common spatial surface. A treatment
must not automatically split the spatial model.

Recommended sequence:

1. Add the required design/interaction and estimability machinery while keeping
   the current single-field models as regression references.
2. Use the fixed-genotype mean/covariance export to implement a staged
   multi-environment analysis. Carry the first-stage covariance as known
   estimation uncertainty, and distinguish it from additional GxE variation.
   Start with a parsimonious genotype/environment model and explicit averaging
   over the target environments. Diagnose shared-genotype connectedness.
3. Add joint plot-level modeling with separate spatial surfaces and residual
   structures per physical field, sharing genotype effects across fields.
   Validate small cases against independent joint-model fits and compare staged
   versus joint results under conditions where they should agree.
4. Add richer genetic covariance (e.g. factor-analytic environment covariance)
   and optional pedigree/genomic relationship matrices after the basic model
   is reliable. Matrix alignment, scaling, PSD and identifiability checks matter.

Outputs stay recognizable: genotype performance overall or by environment,
comparisons with checks in those contexts, and plot results. Any overall mean
must specify its environment weights; extra plots in one environment must not
silently define breeding priorities. Cross-environment summaries should include
observed versus model-predicted support and avoid an undefined universal
"stability" score. Add interpretable response profiles before composite rankings.

Factor-analytic genetic covariance is a substantial modeling extension with
published multi-environment applications; see [Argaw et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12034942/).
It should be a tested optional structure, not the first mandatory complexity
users encounter.

## Proposed development order

Implement the compact output layer first, then build the design support needed
for treatment-specific estimates and multi-environment analysis. Use the same
outputs to assess spline sensitivity and influence/robustness. Develop staged
multi-environment analysis from the mean/covariance export, then joint fitting.
This order supports the most interesting new analyses while preserving a
simple everyday interface and an unchanged reference estimator for comparison.
