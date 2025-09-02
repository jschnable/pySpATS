#!/usr/bin/env bash
set -euo pipefail

# Smoke test
cd dist/spats_py

# Ensure Python can import the local package
export PYTHONPATH=".:${PYTHONPATH:-}"

python3 - <<'PY'
import sys
import pandas as pd
import numpy as np

# show where we're importing from if something goes wrong
# print("\n".join(sys.path))

from spats import SpATS

# Create minimal test data
np.random.seed(42)
data = pd.DataFrame({
    'genotype': ['G1', 'G2', 'G3'] * 10,
    'col': list(range(1, 11)) * 3,
    'row': [1]*10 + [2]*10 + [3]*10,
    'treatment': ['A', 'B'] * 15,
    'yield': np.random.normal(50, 5, 30)
})

# Fit minimal model
model = SpATS(response='yield', genotype='genotype', spatial=('col', 'row'), data=data)
blues = model.get_BLUEs()
assert len(blues) == 3
print('SMOKE TEST OK')
PY
