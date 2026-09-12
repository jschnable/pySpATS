# Moving from the 2025 port

Do not expect the old fitted numbers to remain unchanged: its statistical
model, penalty updates and effective dimensions were incorrect. Validate
new results against the R reference, not the previous Python output.

- Keep `SpATS(response=..., genotype=..., spatial=("col", "row"), data=...)`,
  or use keyword-only `fit_trial`. Constructor fitting remains immediate.
- `PSANOVA` and `SAP` return immutable `SpatialSpec` objects. Both are supported
  by fitting. The tuple shorthand explicitly selects PSANOVA. Use named
  `nseg`, `degree`, `penalty_order`, `nest_div`, and `center` options.
- `summary()` and `summary_ed()` return pandas objects. Use `print(...)` in
  terminal scripts. Monitoring is opt-in.
- `get_heritability(model)` and `model.heritability` require random genotypes.
  The old numeric `get_heritability(ED, count, mode=...)` API is removed.
- Use `genotype_predictions()` for effects and conditional SEs. There is no
  `get_BLUEs()` shortcut returning spatially corrected plot phenotypes.
- `residuals` consistently means response residuals. For Gaussian models it
  equals y-fitted. It is not an undocumented approximation to deviance residuals.
- Use `predict(newdata, offset=..., return_se=True)` for plot predictions.
  New-data offsets default to zero. Unseen levels/extrapolation are errors.
- `plot()` returns four field maps of real fitted quantities. `plot_spats`
  supports `which="spatial"`, `"residuals"`, `"fitted"` or `"all"`.
- `load_wheatdata()` now returns the actual R wheat trial. It is no longer a
  simulated stand-in. Synthetic trials remain explicitly available through
  `generate_field_trial_data` and `create_toy_example`.
- `solver`, `reml`, `spatial`, `basis`, `psanova_basis` and `ed_selected_inverse`
  from the old implementation are retired. They were disconnected/inconsistent
  experimental paths, not alternate supported estimators. Tests specific to
  those internal paths were replaced by checks of the production estimator.
- Python >=3.10. Plotting is optional (`.[plot]`). No CHOLMOD or scikit-learn.

The historical notebook, precomputed sorghum PNGs/CSVs and R summary file in
`examples/` are retained as historical artifacts; they are not current
validation evidence. The current runnable examples are `wheat.py`,
`pyspats_sorghum_example.py`, and `batch_trials.py`.
