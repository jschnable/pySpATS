# Examples

After `pip install -e '.[plot]'`, run from the repository root:

```bash
python examples/wheat.py
python examples/pyspats_sorghum_example.py --output /tmp/pyspats-sorghum
OPENBLAS_NUM_THREADS=1 python examples/batch_trials.py trials.csv results --workers 4
```

The wheat script fits the actual R reference dataset. The sorghum script fits
`EstimatedPlotYield` in the existing CSV, with user-visible convergence and
missingness. The batch script expects trial, yield, height, genotype, col, row;
it exports per-job tables plus a manifest, and exits unsuccessfully if any job
fails or does not converge.

The notebook and preexisting PNG/CSV result files are historical 2025 artifacts.
They do not validate the new implementation. Current cross-language reference
outputs and the regeneration script are in `tests/reference/`.
