# PyMR-PRESSO

A Python implementation of MR-PRESSO for Mendelian randomization analyses.

## Corrected implementation

The core implementation is in \`pymrpresso.py\`. The notebook remains available for interactive use, but import the functions from the module to avoid duplicated implementations.

### Requirements

Python 3.10 or later is recommended.

\`\`\`bash
pip install numpy pandas scipy statsmodels joblib pytest
\`\`\`

### Quick start

Your input data must contain effect estimates and standard errors for the exposure(s) and outcome. Standard errors must be strictly positive. Optional SNP identifiers are preserved when a column named \`SNP\`, \`rsid\`, \`variant_id\`, or \`variant\` is present.

\`\`\`python
import pandas as pd
from pymrpresso import mr_presso, mr_presso_single_optimized

data = pd.read_csv("presso.csv")

# One exposure
result = mr_presso_single_optimized(
    BetaOutcome="Y_effect",
    BetaExposure="E1_effect",
    SdOutcome="Y_se",
    SdExposure="E1_se",
    data=data,
    OUTLIERtest=True,
    DISTORTIONtest=True,
    SignifThreshold=0.05,
    NbDistribution=8000,
    seed=80,
    n_jobs=-1,
)

print(result["Global Test"])
print(result.get("Outlier Test"))
print(result.get("Distortion Test"))
print(result["Main MR results"])
\`\`\`

For multiple exposures, pass lists of column names to \`BetaExposure\` and \`SdExposure\`.

### Statistical and robustness fixes

- Global Test uses leave-one-out (LOO) RSS regardless of whether Outlier Test is enabled.
- Simulated outcomes use observed exposure effects and observed-data LOO causal estimates, consistent with the reference R algorithm.
- Outlier residuals use observed-data LOO estimates, with consistent row/column orientation for multiple exposures.
- Distortion-test resampling samples non-outliers with replacement and uses the reference subset-size convention.
- Bonferroni-adjusted outlier P-value reporting uses the lower bound \`n_snps / NbDistribution\`.
- Optional SNP identifiers and original row indices are preserved in the Outlier Test.
- Parallel simulations use pre-generated per-replicate seeds, so results are reproducible across worker counts for a fixed seed.
- Inputs are validated for missing columns, non-finite values, non-positive standard errors, insufficient SNPs, and invalid simulation counts.

### Tests

Run from the repository root:

\`\`\`bash
pip install pytest
pytest -q
\`\`\`

The tests cover LOO Global Test consistency, single- and multiple-exposure execution, SNP identifier preservation, distortion resampling, empirical P-value formatting, and invalid standard errors.

## Scientific note

This project implements a statistical method; passing software tests does not by itself establish equivalence to every release of the reference R package. Before using results in a publication, compare output against the official [MR-PRESSO R implementation](https://github.com/rondolab/MR-PRESSO/blob/master/R/mr_presso.R) on the same input data and simulation count, and document the software version and random seed.

## Citation

Verbanck, M., Chen, C.-Y., Neale, B., & Do, R. (2018). Detection of widespread horizontal pleiotropy in causal relationships inferred from Mendelian randomization between complex traits and diseases. *Nature Genetics*, 50(5), 693–698. https://doi.org/10.1038/s41588-018-0099-7
