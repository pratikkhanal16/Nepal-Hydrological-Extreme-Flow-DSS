"""
HydroEVT v2: distribution fitting + bootstrap quantile uncertainty

Purpose
-------
Separate two questions that should NOT be conflated:
1) How well does each candidate distribution fit the observed annual extremes?
2) How uncertain are the return-level estimates from that distribution?

Maximum-flow candidates (L-moment based):
    GEV, GUM, KAP, GLO, GPA, LP3
Minimum-flow candidates:
    LP3, LOGN, GAM, WEI, GEV_MIN, GUM_MIN

For each model this script reports:
- GOF: Anderson-Darling statistic, Cramer-von Mises statistic, KS statistic
- AICc evaluated at fitted parameters (reported as supporting information)
- Bootstrap uncertainty for every return period:
    point estimate, bootstrap median, bootstrap SE, 95% CI,
    relative 95% CI half-width (%), bootstrap CV (%), bias, success/failure rate
- Distribution-choice uncertainty across candidate models for each return period.

Installation:
    pip install numpy pandas scipy openpyxl lmoments3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import math
import warnings

import numpy as np
import pandas as pd

from lmoments3 import distr

EPS = 1e-12

MAX_DISTS = ["GEV", "GUM", "KAP", "GLO", "GPA", "LP3"]
MIN_DISTS = ["LP3", "LOGN", "GAM", "WEI", "GEV_MIN", "GUM_MIN"]
DEFAULT_RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200]


@dataclass
class FittedModel:
    name: str
    params: dict
    transformed: bool = False


def _clean(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return x


def _require_positive(x: np.ndarray, model: str):
    if np.any(x <= 0):
        raise ValueError(f"{model} requires strictly positive discharge values.")


def fit_model(values: Iterable[float], model: str) -> FittedModel:
    """Fit one candidate distribution using L-moments.

    LP3 and LOGN are fit in base-10 log space.
    GEV_MIN and GUM_MIN fit -Q as a maximum-type variate and transform back.
    """
    x = _clean(values)
    if len(x) < 5:
        raise ValueError("At least 5 observations are required to fit a model.")

    model = model.upper()

    if model == "GEV":
        return FittedModel(model, dict(distr.gev.lmom_fit(x)))
    if model == "GUM":
        return FittedModel(model, dict(distr.gum.lmom_fit(x)))
    if model == "KAP":
        return FittedModel(model, dict(distr.kap.lmom_fit(x)))
    if model == "GLO":
        return FittedModel(model, dict(distr.glo.lmom_fit(x)))
    if model == "GPA":
        return FittedModel(model, dict(distr.gpa.lmom_fit(x)))
    if model == "PE3":
        return FittedModel(model, dict(distr.pe3.lmom_fit(x)))
    if model == "GAM":
        _require_positive(x, model)
        return FittedModel(model, dict(distr.gam.lmom_fit(x)))
    if model == "WEI":
        _require_positive(x, model)
        return FittedModel(model, dict(distr.wei.lmom_fit(x)))
    if model == "LP3":
        _require_positive(x, model)
        y = np.log10(x)
        return FittedModel(model, dict(distr.pe3.lmom_fit(y)), transformed=True)
    if model == "LOGN":
        _require_positive(x, model)
        y = np.log10(x)
        return FittedModel(model, dict(distr.nor.lmom_fit(y)), transformed=True)
    if model == "GEV_MIN":
        return FittedModel(model, dict(distr.gev.lmom_fit(-x)), transformed=True)
    if model == "GUM_MIN":
        return FittedModel(model, dict(distr.gum.lmom_fit(-x)), transformed=True)

    raise KeyError(f"Unsupported distribution: {model}")


def model_cdf(x: np.ndarray | float, fit: FittedModel) -> np.ndarray:
    q = np.asarray(x, dtype=float)
    m, p = fit.name, fit.params

    if m == "GEV":
        out = distr.gev.cdf(q, **p)
    elif m == "GUM":
        out = distr.gum.cdf(q, **p)
    elif m == "KAP":
        out = distr.kap.cdf(q, **p)
    elif m == "GLO":
        out = distr.glo.cdf(q, **p)
    elif m == "GPA":
        out = distr.gpa.cdf(q, **p)
    elif m == "PE3":
        out = distr.pe3.cdf(q, **p)
    elif m == "GAM":
        out = distr.gam.cdf(q, **p)
    elif m == "WEI":
        out = distr.wei.cdf(q, **p)
    elif m == "LP3":
        out = distr.pe3.cdf(np.log10(q), **p)
    elif m == "LOGN":
        out = distr.nor.cdf(np.log10(q), **p)
    elif m == "GEV_MIN":
        out = 1.0 - distr.gev.cdf(-q, **p)
    elif m == "GUM_MIN":
        out = 1.0 - distr.gum.cdf(-q, **p)
    else:
        raise KeyError(m)

    return np.clip(np.asarray(out, dtype=float), EPS, 1.0 - EPS)


def model_logpdf(x: np.ndarray | float, fit: FittedModel) -> np.ndarray:
    q = np.asarray(x, dtype=float)
    m, p = fit.name, fit.params

    if m == "GEV":
        out = distr.gev.logpdf(q, **p)
    elif m == "GUM":
        out = distr.gum.logpdf(q, **p)
    elif m == "KAP":
        out = distr.kap.logpdf(q, **p)
    elif m == "GLO":
        out = distr.glo.logpdf(q, **p)
    elif m == "GPA":
        out = distr.gpa.logpdf(q, **p)
    elif m == "PE3":
        out = distr.pe3.logpdf(q, **p)
    elif m == "GAM":
        out = distr.gam.logpdf(q, **p)
    elif m == "WEI":
        out = distr.wei.logpdf(q, **p)
    elif m == "LP3":
        _require_positive(q, m)
        # y = log10(q); f_Q(q) = f_Y(y) / (q ln 10)
        out = distr.pe3.logpdf(np.log10(q), **p) - np.log(q * np.log(10.0))
    elif m == "LOGN":
        _require_positive(q, m)
        out = distr.nor.logpdf(np.log10(q), **p) - np.log(q * np.log(10.0))
    elif m == "GEV_MIN":
        out = distr.gev.logpdf(-q, **p)
    elif m == "GUM_MIN":
        out = distr.gum.logpdf(-q, **p)
    else:
        raise KeyError(m)

    return np.asarray(out, dtype=float)


def model_ppf(prob: float | np.ndarray, fit: FittedModel) -> np.ndarray:
    p0 = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    m, par = fit.name, fit.params

    if m == "GEV":
        q = distr.gev.ppf(p0, **par)
    elif m == "GUM":
        q = distr.gum.ppf(p0, **par)
    elif m == "KAP":
        q = distr.kap.ppf(p0, **par)
    elif m == "GLO":
        q = distr.glo.ppf(p0, **par)
    elif m == "GPA":
        q = distr.gpa.ppf(p0, **par)
    elif m == "PE3":
        q = distr.pe3.ppf(p0, **par)
    elif m == "GAM":
        q = distr.gam.ppf(p0, **par)
    elif m == "WEI":
        q = distr.wei.ppf(p0, **par)
    elif m == "LP3":
        q = 10.0 ** distr.pe3.ppf(p0, **par)
    elif m == "LOGN":
        q = 10.0 ** distr.nor.ppf(p0, **par)
    elif m == "GEV_MIN":
        q = -distr.gev.ppf(1.0 - p0, **par)
    elif m == "GUM_MIN":
        q = -distr.gum.ppf(1.0 - p0, **par)
    else:
        raise KeyError(m)

    return np.asarray(q, dtype=float)


def return_level(fit: FittedModel, T: float, analysis: str) -> float:
    analysis = analysis.lower()
    if T <= 1:
        raise ValueError("Return period T must be > 1.")
    # Maximum: annual exceedance probability = 1/T => CDF = 1 - 1/T
    # Minimum: annual nonexceedance probability = 1/T => CDF = 1/T
    p = 1.0 - 1.0 / T if analysis == "maximum" else 1.0 / T
    q = float(model_ppf(p, fit))
    if not np.isfinite(q):
        raise ValueError("Non-finite return level")
    return q


def gof_statistics(values: Iterable[float], fit: FittedModel) -> Dict[str, float]:
    """Tail-sensitive / distributional GOF statistics; lower is better."""
    x = np.sort(_clean(values))
    n = len(x)
    F = model_cdf(x, fit)
    i = np.arange(1, n + 1, dtype=float)

    # Kolmogorov-Smirnov statistic (statistic only; fitted-parameter p-value omitted)
    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1.0) / n)
    ks = max(d_plus, d_minus)

    # Cramer-von Mises
    cvm = 1.0 / (12.0 * n) + np.sum((F - (2.0 * i - 1.0) / (2.0 * n)) ** 2)

    # Anderson-Darling; gives more weight to tails than KS/CvM
    Fr = F[::-1]
    ad = -n - np.mean((2.0 * i - 1.0) * (np.log(F) + np.log(1.0 - Fr)))

    # AICc at fitted parameters (supporting metric; all models fitted with L-moments here)
    logpdf = model_logpdf(x, fit)
    if np.any(~np.isfinite(logpdf)):
        loglik = -np.inf
        aicc = np.inf
    else:
        loglik = float(np.sum(logpdf))
        k = len(fit.params)
        aic = 2.0 * k - 2.0 * loglik
        aicc = aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0) if n > k + 1 else np.inf

    return {"AD": float(ad), "CvM": float(cvm), "KS": float(ks), "AICc": float(aicc)}


def bootstrap_quantiles(
    values: Iterable[float],
    model: str,
    analysis: str,
    return_periods: Iterable[int] = DEFAULT_RETURN_PERIODS,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Nonparametric bootstrap: resample observed years with replacement, refit model,
    and recompute all requested return levels.
    """
    x = _clean(values)
    n = len(x)
    rps = [int(T) for T in return_periods]
    rng = np.random.default_rng(seed)

    fit0 = fit_model(x, model)
    point = {T: return_level(fit0, T, analysis) for T in rps}

    boot = {T: [] for T in rps}
    failed = 0

    for _ in range(int(n_boot)):
        xb = rng.choice(x, size=n, replace=True)
        try:
            fb = fit_model(xb, model)
            vals = {T: return_level(fb, T, analysis) for T in rps}

            # Annual minimum discharge should not be negative. Treat such extrapolations
            # as unstable/invalid for low-flow inference.
            if analysis.lower() == "minimum" and any(v < 0 for v in vals.values()):
                raise ValueError("Negative minimum-flow return level")

            for T, v in vals.items():
                if not np.isfinite(v):
                    raise ValueError("Non-finite bootstrap return level")
            for T, v in vals.items():
                boot[T].append(float(v))
        except Exception:
            failed += 1

    success = n_boot - failed
    success_rate = success / n_boot if n_boot else np.nan
    rows = []

    for T in rps:
        arr = np.asarray(boot[T], dtype=float)
        if len(arr) < max(50, int(0.5 * n_boot)):
            lo = med = hi = se = meanb = bias = cv = rel_half = np.nan
        else:
            lo, med, hi = np.percentile(arr, [2.5, 50.0, 97.5])
            se = float(np.std(arr, ddof=1))
            meanb = float(np.mean(arr))
            bias = meanb - point[T]
            cv = 100.0 * se / abs(meanb) if meanb != 0 else np.nan
            rel_half = 100.0 * (hi - lo) / (2.0 * abs(point[T])) if point[T] != 0 else np.nan

        rows.append({
            "Distribution": model,
            "Analysis": analysis.lower(),
            "Return_Period": T,
            "Point_Estimate": point[T],
            "Bootstrap_Median": med,
            "Bootstrap_SE": se,
            "CI95_Lower": lo,
            "CI95_Upper": hi,
            "CI95_Width": hi - lo if np.isfinite(hi) and np.isfinite(lo) else np.nan,
            "Relative_CI_HalfWidth_pct": rel_half,
            "Bootstrap_CV_pct": cv,
            "Bootstrap_Bias": bias,
            "Bootstrap_Success_pct": 100.0 * success_rate,
            "Bootstrap_Failure_pct": 100.0 * (1.0 - success_rate),
            "N": n,
            "N_Boot": int(n_boot),
        })

    return pd.DataFrame(rows)


def analyze_station(
    values: Iterable[float],
    analysis: str,
    return_periods: Iterable[int] = DEFAULT_RETURN_PERIODS,
    n_boot: int = 2000,
    seed: int = 42,
    distributions: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (fit_ranking, quantile_uncertainty, model_choice_uncertainty)."""
    x = _clean(values)
    analysis = analysis.lower()
    if distributions is None:
        distributions = MAX_DISTS if analysis == "maximum" else MIN_DISTS

    fit_rows = []
    unc_frames = []

    for j, model in enumerate(distributions):
        try:
            fit = fit_model(x, model)
            g = gof_statistics(x, fit)
            fit_rows.append({"Distribution": model, "Analysis": analysis, "N": len(x), **g})
            unc = bootstrap_quantiles(
                x, model, analysis,
                return_periods=return_periods,
                n_boot=n_boot,
                seed=seed + 1009 * j,
            )
            unc_frames.append(unc)
        except Exception as e:
            fit_rows.append({
                "Distribution": model, "Analysis": analysis, "N": len(x),
                "AD": np.nan, "CvM": np.nan, "KS": np.nan, "AICc": np.nan,
                "Fit_Error": str(e),
            })

    fit_df = pd.DataFrame(fit_rows)
    if fit_df.empty:
        return fit_df, pd.DataFrame(), pd.DataFrame()

    # Fit ranking: do NOT use RMSE/MAE. Keep GOF and uncertainty conceptually separate.
    for c in ["AD", "CvM", "KS"]:
        fit_df[f"Rank_{c}"] = fit_df[c].rank(method="min", ascending=True, na_option="bottom")
    fit_df["Fit_Rank_Sum"] = fit_df[["Rank_AD", "Rank_CvM", "Rank_KS"]].sum(axis=1)
    fit_df["Fit_Rank"] = fit_df["Fit_Rank_Sum"].rank(method="min", ascending=True, na_option="bottom").astype("Int64")

    unc_df = pd.concat(unc_frames, ignore_index=True) if unc_frames else pd.DataFrame()
    if not unc_df.empty:
        # Join fit rank and calculate uncertainty rank separately at each return period.
        unc_df = unc_df.merge(fit_df[["Distribution", "Fit_Rank", "Fit_Rank_Sum"]], on="Distribution", how="left")
        unc_df["Uncertainty_Rank"] = unc_df.groupby("Return_Period")["Relative_CI_HalfWidth_pct"].rank(
            method="min", ascending=True, na_option="bottom"
        ).astype("Int64")

        # Summary uncertainty by model = median relative CI half-width across requested T values.
        u_summary = (
            unc_df.groupby("Distribution", as_index=False)
            .agg(
                Median_Relative_CI_HalfWidth_pct=("Relative_CI_HalfWidth_pct", "median"),
                Max_Relative_CI_HalfWidth_pct=("Relative_CI_HalfWidth_pct", "max"),
                Mean_Bootstrap_Failure_pct=("Bootstrap_Failure_pct", "mean"),
            )
        )
        u_summary["Overall_Uncertainty_Rank"] = u_summary["Median_Relative_CI_HalfWidth_pct"].rank(
            method="min", ascending=True, na_option="bottom"
        ).astype("Int64")
        fit_df = fit_df.merge(u_summary, on="Distribution", how="left")

    # Distribution-choice (epistemic/model) uncertainty at each T.
    model_rows = []
    if not unc_df.empty:
        top3 = set(fit_df.nsmallest(3, "Fit_Rank_Sum")["Distribution"])
        for T, g in unc_df.groupby("Return_Period"):
            q = pd.to_numeric(g["Point_Estimate"], errors="coerce").dropna().values
            gt = g[g["Distribution"].isin(top3)]
            qt = pd.to_numeric(gt["Point_Estimate"], errors="coerce").dropna().values
            model_rows.append({
                "Analysis": analysis,
                "Return_Period": int(T),
                "All_Model_Min": np.min(q) if len(q) else np.nan,
                "All_Model_Max": np.max(q) if len(q) else np.nan,
                "All_Model_Range": np.ptp(q) if len(q) else np.nan,
                "All_Model_SD": np.std(q, ddof=1) if len(q) > 1 else np.nan,
                "Top3_Model_Min": np.min(qt) if len(qt) else np.nan,
                "Top3_Model_Max": np.max(qt) if len(qt) else np.nan,
                "Top3_Model_Range": np.ptp(qt) if len(qt) else np.nan,
                "Top3_Distributions": ", ".join(sorted(top3)),
            })
    model_df = pd.DataFrame(model_rows)

    return fit_df.sort_values(["Fit_Rank_Sum", "AD"], na_position="last"), unc_df, model_df


def _find_sheet(sheet_names: List[str], candidates: List[str]) -> str | None:
    lower = {s.lower(): s for s in sheet_names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def build_workbook(
    annual_excel: str,
    output_excel: str = "Nepal_HydroEVT_Uncertainty_v2.xlsx",
    return_periods: Iterable[int] = DEFAULT_RETURN_PERIODS,
    n_boot: int = 2000,
    seed: int = 42,
    min_valid_years: int = 10,
):
    """Run all station columns from Maximum/Minimum sheets and save results.

    WARNING: 2000 bootstraps x many stations x many distributions can take substantial time.
    Use n_boot=300 or 500 while debugging, then n_boot=2000 for final analysis.
    """
    xls = pd.ExcelFile(annual_excel)
    max_sheet = _find_sheet(xls.sheet_names, ["Maximum", "maximum"])
    min_sheet = _find_sheet(xls.sheet_names, ["Minimum", "minimum"])

    if not max_sheet and not min_sheet:
        raise ValueError("Could not find Maximum or Minimum sheets.")

    all_fit, all_unc, all_model = [], [], []

    for analysis, sheet in [("maximum", max_sheet), ("minimum", min_sheet)]:
        if sheet is None:
            continue
        df = pd.read_excel(annual_excel, sheet_name=sheet)
        if "Year" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Year"})

        station_cols = [c for c in df.columns if str(c).strip().lower() != "year"]
        for si, station in enumerate(station_cols):
            vals = pd.to_numeric(df[station], errors="coerce").dropna().values
            if len(vals) < min_valid_years:
                continue

            print(f"[{analysis.upper()}] Station {station}: n={len(vals)}")
            fit_df, unc_df, model_df = analyze_station(
                vals,
                analysis=analysis,
                return_periods=return_periods,
                n_boot=n_boot,
                seed=seed + 100000 * si + (0 if analysis == "maximum" else 50000),
            )
            for d in [fit_df, unc_df, model_df]:
                if not d.empty:
                    d.insert(0, "Station", str(station))
            all_fit.append(fit_df)
            all_unc.append(unc_df)
            all_model.append(model_df)

    fit_all = pd.concat(all_fit, ignore_index=True) if all_fit else pd.DataFrame()
    unc_all = pd.concat(all_unc, ignore_index=True) if all_unc else pd.DataFrame()
    model_all = pd.concat(all_model, ignore_index=True) if all_model else pd.DataFrame()

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        if not fit_all.empty:
            fit_all[fit_all["Analysis"] == "maximum"].to_excel(writer, "MAX_Model_Fit_Uncertainty", index=False)
            fit_all[fit_all["Analysis"] == "minimum"].to_excel(writer, "MIN_Model_Fit_Uncertainty", index=False)
        if not unc_all.empty:
            unc_all[unc_all["Analysis"] == "maximum"].to_excel(writer, "MAX_Quantile_Bootstrap", index=False)
            unc_all[unc_all["Analysis"] == "minimum"].to_excel(writer, "MIN_Quantile_Bootstrap", index=False)
        if not model_all.empty:
            model_all[model_all["Analysis"] == "maximum"].to_excel(writer, "MAX_Distribution_Choice", index=False)
            model_all[model_all["Analysis"] == "minimum"].to_excel(writer, "MIN_Distribution_Choice", index=False)

    print(f"Saved: {output_excel}")
    return fit_all, unc_all, model_all


if __name__ == "__main__":
    # Edit these paths as needed.
    build_workbook(
        annual_excel="Maximum Yearly Discharge.xlsx",
        output_excel="Nepal_HydroEVT_Uncertainty_v2.xlsx",
        n_boot=500,  # use 2000 for final analysis after testing
    )
