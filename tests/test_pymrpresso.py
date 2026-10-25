import numpy as np
import pandas as pd
import pytest

from pymrpresso import (
    _draw_distortion_subset,
    _format_zero_pvalue,
    mr_presso,
    mr_presso_single_optimized,
)


def make_data(seed=5, n=28, multi=False, outlier=True):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.15, 0.08, n)
    x2 = rng.normal(0.05, 0.09, n)
    y = 0.6 * x1 + (0.2 * x2 if multi else 0) + rng.normal(0, 0.035, n)
    if outlier:
        y[7] += 0.5
    values = {
        "rsid": [f"rs{i}" for i in range(n)],
        "x1": x1,
        "sx1": np.full(n, 0.025),
        "y": y,
        "sy": np.full(n, 0.03),
    }
    if multi:
        values.update({"x2": x2, "sx2": np.full(n, 0.03)})
    return pd.DataFrame(values, index=np.arange(100, 100 + n))


def test_global_test_uses_loo_rss_with_or_without_outlier_test():
    data = make_data()
    without = mr_presso("y", "x1", "sy", "sx1", data, NbDistribution=100, seed=123)
    with_outlier = mr_presso(
        "y", "x1", "sy", "sx1", data,
        OUTLIERtest=True, NbDistribution=100, seed=123,
    )
    assert without["Global Test"]["RSSobs"] == pytest.approx(
        with_outlier["Global Test"]["RSSobs"]
    )
    assert without["Global Test"]["Pvalue"] == with_outlier["Global Test"]["Pvalue"]


def test_multiexposure_outlier_test_preserves_snp_ids_and_correct_order():
    data = make_data(multi=True)
    result = mr_presso(
        "y", ["x1", "x2"], "sy", ["sx1", "sx2"], data,
        OUTLIERtest=True, DISTORTIONtest=True, NbDistribution=100, seed=123,
    )
    assert "Outlier Test" in result
    assert list(result["Outlier Test"]["SNP"]) == list(data["rsid"])
    mr = result["Main MR results"]
    assert list(mr["MR Analysis"]) == ["Raw", "Raw", "Outlier-corrected", "Outlier-corrected"]
    assert list(mr["Exposure"]) == ["x1", "x2", "x1", "x2"]


def test_optimized_and_general_single_exposure_paths_match_when_serial():
    data = make_data()
    general = mr_presso(
        "y", "x1", "sy", "sx1", data,
        OUTLIERtest=True, DISTORTIONtest=True, NbDistribution=100, seed=123,
    )
    optimized = mr_presso_single_optimized(
        "y", "x1", "sy", "sx1", data,
        OUTLIERtest=True, DISTORTIONtest=True, NbDistribution=100, seed=123, n_jobs=1,
    )
    assert general["Global Test"] == optimized["Global Test"]
    pd.testing.assert_frame_equal(general["Outlier Test"], optimized["Outlier Test"])
    assert general["Distortion Test"]["Outliers Indices"] == optimized["Distortion Test"]["Outliers Indices"]


def test_distortion_sampling_uses_expected_subset_size_and_varies():
    rng = np.random.default_rng(7)
    outliers = np.array([1, 4])
    first = _draw_distortion_subset(rng, 10, outliers)
    second = _draw_distortion_subset(rng, 10, outliers)
    assert len(first) == 8
    assert len(second) == 8
    assert not np.array_equal(first, second)


def test_zero_empirical_pvalue_uses_supplied_reporting_bound():
    assert _format_zero_pvalue(0.0, 30 / 8000) == "<0.00375"
    assert _format_zero_pvalue(0.01, 1 / 100) == pytest.approx(0.01)


def test_rejects_nonpositive_standard_errors():
    data = make_data(outlier=False)
    data.loc[data.index[0], "sy"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        mr_presso("y", "x1", "sy", "sx1", data, NbDistribution=100, seed=1)
