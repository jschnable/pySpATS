# Timing protocol

From the repository root, after installing the package:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python benchmarks/benchmark.py
```

The deterministic generator produces complete Gaussian trials with replicated
genotypes, additive spatial trends and noise. Timings cover basis construction,
fitting, effective dimensions and final coefficient covariance; they exclude
CSV I/O. Resolution is nseg=(10,10), nest_div=2, tolerance=1e-6, max_iter=500.
Each configuration is timed once; these are local observations, not a hardware
independent guarantee or a statistically rigorous performance study.

Observed on 2026-09-12, macOS arm64, Python 3.14.7, NumPy 2.5.1, SciPy 1.18.0:

| Plots | Genotypes | Fixed genotype, seconds | Random genotype, seconds |
|---:|---:|---:|---:|
| 192 | 24 | 0.0052 | 0.0040 |
| 2,000 | 400 | 0.0144 | 0.0136 |
| 10,000 | 2,000 | 0.3311 | 0.4598 |

All six converged. On the same 2,000-plot CSV, R SpATS 1.0-20 took 0.138 s
(fixed) and 0.145 s (random), both 28 SAP iterations; Python also took 28.
R was called with OPENBLAS_NUM_THREADS=1 and timed with system.time around
SpATS after data loading. See compare_r.R. The roughly tenfold ratio is only
for this particular configuration; no claim is made for every model or field.

To reproduce the R comparison:

```bash
PYTHONPATH=. python -c "from benchmarks.benchmark import data; data(50,40,400).to_csv('/tmp/pyspats-benchmark.csv',index=False)"
OPENBLAS_NUM_THREADS=1 Rscript benchmarks/compare_r.R /tmp/pyspats-benchmark.csv
```

The design and final covariance are still dense. The 10,000-plot, 2,000-genotype
case alone requires roughly 160 MB for one n-by-g float64 indicator array;
several arrays coexist during fitting. This is not a low-memory distributed
solver. Increasing nseg in both axes increases the interaction dimension
multiplicatively. Use explicit nested bases and allocate memory per worker.

## Matched 10,000-plot comparison for 0.3.0

On the same Apple M5 machine, both implementations read the identical
10,000-plot, 2,000-genotype CSV. Three sequential fits per configuration used
nseg=(10,10), nest_div=2, tolerance=1e-6, max_iter=500. Library imports and CSV
I/O are outside the timers; full fitting is inside. Garbage collection occurs
before each run. All BLAS/OpenMP thread limits were set to one.

| Genotype model | Python median (range), seconds | R median (range), seconds | R/Python median ratio | Iterations, both |
|---|---:|---:|---:|---:|
| Fixed | 0.390 (0.378–0.568) | 1.385 (1.129–1.812) | 3.55× | 71 |
| Random | 0.576 (0.566–0.652) | 6.091 (5.934–6.271) | 10.57× | 121 |

The maximum absolute fitted-value difference was 1.12e-10 (fixed) and
6.40e-11 (random). Thus the runtime comparison uses effectively identical
answers and the same iteration counts. It is a local benchmark, not a general
speed guarantee. The earlier single Python timings above are retained as the
initial measurements; use these matched-run medians for the R comparison.

Raw measurements: [timings](results/10000_timings.csv) and
[environment/settings/agreement](results/10000_metadata.json).
R 4.6.1 with SpATS 1.0-20; Python 3.14.7 with pySpATS 0.3.0.

Reproduce (requires the supplied R package):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python benchmarks/compare.py --output /tmp/spats-compare --repeats 3
```
