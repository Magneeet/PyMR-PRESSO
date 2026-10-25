"""Python implementation of MR-PRESSO, following the reference R algorithm.

The public functions preserve the argument names used by the original
PyMR-PRESSO notebook. Both observed and simulated RSS use leave-one-out (LOO)
regression; the optimized entry point parallelizes null simulations.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed


_SNP_ID_COLUMNS = ("SNP", "snp", "rsid", "RSID", "variant_id", "variant")


def _as_names(value: str | Sequence[str], name: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    if not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{name} must be a column name or a non-empty sequence of column names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicate column names")
    return values


def _loo_rss_and_betas(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return LOO RSS and one p-dimensional LOO coefficient vector per SNP."""
    n, p = x.shape
    if n <= p:
        raise ValueError("Not enough SNPs to fit leave-one-out regressions")

    sqrt_w = np.sqrt(weights)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    beta_loo = np.empty((n, p), dtype=float)

    for i in range(n):
        keep = np.arange(n) != i
        x_loo = xw[keep]
        y_loo = yw[keep]
        if np.linalg.matrix_rank(x_loo) < p:
            raise ValueError(
                f"Leave-one-out design matrix is rank deficient after removing row {i}; "
                "check for collinear exposures or too few independent SNPs."
            )
        beta_loo[i], *_ = np.linalg.lstsq(x_loo, y_loo, rcond=None)

    residuals = yw - np.einsum("ij,ij->i", xw, beta_loo)
    return float(np.sum(residuals**2)), beta_loo


def _simulate_one(
    seed: int,
    x: np.ndarray,
    sd_x: np.ndarray,
    sd_y: np.ndarray,
    weights: np.ndarray,
    beta_loo_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate one MR-PRESSO null dataset and calculate its LOO RSS."""
    rng = np.random.default_rng(seed)
    x_sim = rng.normal(loc=x, scale=sd_x)

    # Reference MR-PRESSO predicts simulated outcomes from the observed
    # exposure effects and each SNP's observed-data LOO causal estimate.
    mean_y_sim = np.einsum("ij,ij->i", x, beta_loo_observed)
    y_sim = rng.normal(loc=mean_y_sim, scale=sd_y)
    rss_sim, _ = _loo_rss_and_betas(x_sim, y_sim, weights)
    return x_sim, y_sim, rss_sim


def _format_zero_pvalue(pvalue: float, lower_bound: float) -> float | str:
    return f"<{lower_bound:g}" if pvalue == 0 else float(pvalue)


def _draw_distortion_subset(
    rng: np.random.Generator,
    n_snps: int,
    outlier_positions: np.ndarray,
) -> np.ndarray:
    """Draw a distortion-test subset using the reference R sampling scheme."""
    k = len(outlier_positions)
    non_outliers = np.setdiff1d(np.arange(n_snps), outlier_positions)
    if k == 0 or k >= n_snps or non_outliers.size == 0:
        raise ValueError("Distortion subset requires at least one outlier and one non-outlier")
    sampled = rng.choice(non_outliers, size=n_snps - k, replace=True)
    combined = np.concatenate([outlier_positions, sampled])
    # The reference implementation uses the first n-k indices.
    return combined[: n_snps - k]


def _fit_wls(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    if np.linalg.matrix_rank(x * np.sqrt(weights)[:, None]) < x.shape[1]:
        raise ValueError("The weighted exposure design matrix is rank deficient")
    return sm.WLS(y, x, weights=weights).fit()


def _run_mr_presso(
    BetaOutcome: str,
    BetaExposure: str | Sequence[str],
    SdOutcome: str,
    SdExposure: str | Sequence[str],
    data: pd.DataFrame,
    OUTLIERtest: bool = False,
    DISTORTIONtest: bool = False,
    SignifThreshold: float = 0.05,
    NbDistribution: int = 1000,
    seed: int | None = None,
    n_jobs: int = 1,
    single_output: bool = False,
) -> dict[str, Any]:
    exposures = _as_names(BetaExposure, "BetaExposure")
    exposure_sds = _as_names(SdExposure, "SdExposure")
    if len(exposures) != len(exposure_sds):
        raise ValueError("BetaExposure and SdExposure must have the same number of elements")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not np.isfinite(SignifThreshold) or not 0 <= SignifThreshold <= 1:
        raise ValueError("SignifThreshold must be between 0 and 1")
    if isinstance(NbDistribution, bool) or int(NbDistribution) != NbDistribution or NbDistribution < 1:
        raise ValueError("NbDistribution must be a positive integer")
    NbDistribution = int(NbDistribution)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")

    required = [BetaOutcome, *exposures, SdOutcome, *exposure_sds]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required data columns: {missing}")

    valid = data.loc[:, required].notna().all(axis=1)
    source_indices = data.index[valid].to_numpy()
    id_column = next((col for col in _SNP_ID_COLUMNS if col in data.columns), None)
    snp_ids = data.loc[valid, id_column].to_numpy() if id_column else source_indices.copy()
    clean = data.loc[valid, required].copy().reset_index(drop=True)
    if clean.empty:
        raise ValueError("No complete observations remain after removing rows with missing values")
    try:
        clean = clean.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All effect estimates and standard errors must be numeric") from exc
    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Effect estimates and standard errors must be finite")

    y = clean[BetaOutcome].to_numpy(dtype=float)
    x = clean[exposures].to_numpy(dtype=float)
    sd_y = clean[SdOutcome].to_numpy(dtype=float)
    sd_x = clean[exposure_sds].to_numpy(dtype=float)
    if np.any(sd_y <= 0) or np.any(sd_x <= 0):
        raise ValueError("Outcome and exposure standard errors must be strictly positive")

    # Orient SNP effects consistently with the first exposure. An exactly-zero
    # first exposure does not turn an otherwise valid row into an all-zero row.
    signs = np.where(x[:, 0] < 0, -1.0, 1.0)
    x = x * signs[:, None]
    y = y * signs
    n_snps, n_exposures = x.shape
    if n_snps <= n_exposures + 2:
        raise ValueError("Not enough instrumental variables")
    if n_snps >= NbDistribution:
        raise ValueError("NbDistribution must be greater than the number of complete SNPs")

    weights = 1.0 / np.square(sd_y)
    rss_observed, beta_loo_observed = _loo_rss_and_betas(x, y, weights)

    # Generate per-replicate seeds in the parent process so results are
    # reproducible regardless of the joblib backend or number of workers.
    seed_rng = np.random.default_rng(seed)
    max_seed = np.iinfo(np.uint32).max
    simulation_seeds = seed_rng.integers(0, max_seed, size=NbDistribution, dtype=np.uint32)
    simulation_args = (x, sd_x, sd_y, weights, beta_loo_observed)
    if n_jobs == 1:
        simulations = [
            _simulate_one(int(sim_seed), *simulation_args) for sim_seed in simulation_seeds
        ]
    else:
        simulations = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_one)(int(sim_seed), *simulation_args)
            for sim_seed in simulation_seeds
        )

    rss_expected = np.fromiter((sim[2] for sim in simulations), dtype=float, count=NbDistribution)
    global_p_raw = float(np.mean(rss_expected > rss_observed))
    global_test = {
        "RSSobs": rss_observed,
        "Pvalue": _format_zero_pvalue(global_p_raw, 1.0 / NbDistribution),
    }

    outlier_test = None
    outlier_positions = np.array([], dtype=int)
    outlier_active = bool(OUTLIERtest and global_p_raw < SignifThreshold)
    if outlier_active:
        observed_dif = y - np.einsum("ij,ij->i", x, beta_loo_observed)
        simulated_dif = np.empty((NbDistribution, n_snps), dtype=float)
        for i, (x_sim, y_sim, _) in enumerate(simulations):
            # Use observed-data LOO coefficients, not estimates refitted on
            # each simulated dataset, for the per-SNP null residuals.
            simulated_dif[i] = y_sim - np.einsum("ij,ij->i", x_sim, beta_loo_observed)
        p_uncorrected = np.mean(
            np.square(simulated_dif) > np.square(observed_dif)[None, :], axis=0
        )
        p_corrected = np.minimum(p_uncorrected * n_snps, 1.0)
        outlier_positions = np.flatnonzero(p_corrected <= SignifThreshold)
        display_p = [
            _format_zero_pvalue(float(p), n_snps / NbDistribution)
            for p in p_corrected
        ]
        outlier_test = pd.DataFrame(
            {"SNP": snp_ids, "RSSobs": np.square(observed_dif), "Pvalue": display_p},
            index=source_indices,
        )

    model_all = _fit_wls(x, y, weights)
    raw_params = np.asarray(model_all.params, dtype=float)
    bias_test: dict[str, Any] | None = None
    corrected_model = None

    if DISTORTIONtest and outlier_active:
        k = len(outlier_positions)
        outlier_source_indices = source_indices[outlier_positions].tolist()
        if k == 0:
            bias_test = {
                "Outliers Indices": "No significant outliers",
                "Distortion Coefficient (%)": None,
                "Pvalue": None,
            }
        elif k >= n_snps:
            bias_test = {
                "Outliers Indices": "All SNPs considered as outliers",
                "Distortion Coefficient (%)": None,
                "Pvalue": None,
            }
        elif n_snps - k <= n_exposures:
            bias_test = {
                "Outliers Indices": outlier_source_indices,
                "Distortion Coefficient (%)": None,
                "Pvalue": None,
                "Warning": "Too few non-outlier SNPs to fit the corrected MR model",
            }
        else:
            keep = np.ones(n_snps, dtype=bool)
            keep[outlier_positions] = False
            corrected_model = _fit_wls(x[keep], y[keep], weights[keep])
            corrected_params = np.asarray(corrected_model.params, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                bias_observed = (raw_params - corrected_params) / np.abs(corrected_params)

            bias_simulated = np.full((NbDistribution, n_exposures), np.nan, dtype=float)
            bias_rng = np.random.default_rng(
                int(seed_rng.integers(0, max_seed, dtype=np.uint32))
            )
            for i in range(NbDistribution):
                subset = _draw_distortion_subset(bias_rng, n_snps, outlier_positions)
                model_random = _fit_wls(x[subset], y[subset], weights[subset])
                random_params = np.asarray(model_random.params, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    bias_simulated[i] = (raw_params - random_params) / np.abs(random_params)

            distortion_p = np.full(n_exposures, np.nan, dtype=float)
            for j in range(n_exposures):
                valid_bias = np.isfinite(bias_simulated[:, j])
                if np.isfinite(bias_observed[j]) and valid_bias.any():
                    distortion_p[j] = np.mean(
                        np.abs(bias_simulated[valid_bias, j]) > abs(bias_observed[j])
                    )
            formatted_distortion_p = [
                _format_zero_pvalue(float(p), 1.0 / NbDistribution) if np.isfinite(p) else np.nan
                for p in distortion_p
            ]
            bias_test = {
                "Outliers Indices": outlier_source_indices,
                "Distortion Coefficient (%)": 100.0 * bias_observed,
                "Pvalue": formatted_distortion_p,
            }

    se_label = "Sd" if single_output else "Std Error"
    raw_rows: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    raw_se = np.asarray(model_all.bse, dtype=float)
    raw_t = np.asarray(model_all.tvalues, dtype=float)
    raw_p = np.asarray(model_all.pvalues, dtype=float)
    if corrected_model is not None:
        corrected_params = np.asarray(corrected_model.params, dtype=float)
        corrected_se = np.asarray(corrected_model.bse, dtype=float)
        corrected_t = np.asarray(corrected_model.tvalues, dtype=float)
        corrected_p = np.asarray(corrected_model.pvalues, dtype=float)
    for j, exposure in enumerate(exposures):
        raw_rows.append({
            "Exposure": exposure,
            "MR Analysis": "Raw",
            "Causal Estimate": raw_params[j],
            se_label: raw_se[j],
            "T-stat": raw_t[j],
            "P-value": raw_p[j],
        })
        if corrected_model is not None:
            corrected_rows.append({
                "Exposure": exposure,
                "MR Analysis": "Outlier-corrected",
                "Causal Estimate": corrected_params[j],
                se_label: corrected_se[j],
                "T-stat": corrected_t[j],
                "P-value": corrected_p[j],
            })
        else:
            corrected_rows.append({
                "Exposure": exposure,
                "MR Analysis": "Outlier-corrected",
                "Causal Estimate": np.nan,
                se_label: np.nan,
                "T-stat": np.nan,
                "P-value": np.nan,
            })
    rows = raw_rows + corrected_rows

    results: dict[str, Any] = {"Global Test": global_test}
    if outlier_active and outlier_test is not None:
        results["Outlier Test"] = outlier_test
    if DISTORTIONtest and outlier_active and bias_test is not None:
        if single_output and isinstance(bias_test.get("Pvalue"), list):
            if len(bias_test["Pvalue"]) == 1:
                bias_test["Pvalue"] = bias_test["Pvalue"][0]
            coeff = bias_test.get("Distortion Coefficient (%)")
            if isinstance(coeff, np.ndarray) and coeff.size == 1:
                bias_test["Distortion Coefficient (%)"] = float(coeff[0])
        results["Distortion Test"] = bias_test
    results["Main MR results"] = pd.DataFrame(rows)

    if outlier_active and n_snps / NbDistribution > SignifThreshold:
        warnings.warn(
            "Outlier test unstable: the minimum Bonferroni-corrected empirical P-value "
            f"is {n_snps / NbDistribution:g}, above SignifThreshold={SignifThreshold}. "
            "Increase NbDistribution.",
            RuntimeWarning,
            stacklevel=2,
        )
    return results


def mr_presso(
    BetaOutcome: str,
    BetaExposure: str | Sequence[str],
    SdOutcome: str,
    SdExposure: str | Sequence[str],
    data: pd.DataFrame,
    OUTLIERtest: bool = False,
    DISTORTIONtest: bool = False,
    SignifThreshold: float = 0.05,
    NbDistribution: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run MR-PRESSO for one or more exposures."""
    return _run_mr_presso(
        BetaOutcome, BetaExposure, SdOutcome, SdExposure, data,
        OUTLIERtest, DISTORTIONtest, SignifThreshold, NbDistribution,
        seed, n_jobs=1, single_output=False,
    )


def mr_presso_single_optimized(
    BetaOutcome: str,
    BetaExposure: str | Sequence[str],
    SdOutcome: str,
    SdExposure: str | Sequence[str],
    data: pd.DataFrame,
    OUTLIERtest: bool = False,
    DISTORTIONtest: bool = False,
    SignifThreshold: float = 0.05,
    NbDistribution: int = 1000,
    seed: int | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run MR-PRESSO with parallelized null simulations.

    The historical name is retained for notebook compatibility. The shared
    implementation also handles multiple exposures correctly.
    """
    return _run_mr_presso(
        BetaOutcome, BetaExposure, SdOutcome, SdExposure, data,
        OUTLIERtest, DISTORTIONtest, SignifThreshold, NbDistribution,
        seed, n_jobs=n_jobs, single_output=True,
    )
