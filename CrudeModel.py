"""Beginner-friendly pipeline to predict pipeline revenues and forecast top companies.

What it does, in simple steps:
- Loads the East Daley CSV
- Extracts a tidy table with one row per (company, year)
- Trains a small model and evaluates it on a test set
- Saves test predictions and feature importance
- Forecasts 2025–2027 and saves top companies per year
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# Row numbers for the six required data types in the input file
ROW_MAP: Dict[str, int] = {
    "target_revenue": 100,           # Total Interstate Operating Revenues (Row 100)
    "opex": 10,                      # Operating and Maintenance Expenses (Row 10)
    "rate_base_primary": 50,         # Rate Base (Row 50)
    "rate_base_fallback": 54,        # Total Rate Base - Trended Original Cost (Row 54)
    "rate_of_return_pct": 60,        # Rate of Return % (Row 60)
    "total_cost_of_service": 90,     # Total Cost of Service (Row 90)
    "throughput_barrels": 110,       # Total Interstate Throughput in Barrels (Row 110)
}


def load_csv(csv_path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Fallback: let pandas guess
    return pd.read_csv(csv_path)


def get_year_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("Q4")]


def tidy_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    year_cols = get_year_columns(raw_df)

    def v(row_df: pd.DataFrame, col: str) -> float:
        if row_df.empty or col not in row_df.columns:
            return np.nan
        return pd.to_numeric(row_df[col].iloc[0], errors="coerce")

    rows: List[Dict[str, float]] = []
    for company, g in raw_df.groupby("Pipe Name"):
        ownership = str(g["Pipe Ownership"].iloc[0]) if "Pipe Ownership" in g.columns else ""

        sel = {
            "target_revenue": g[g["Row Number"] == ROW_MAP["target_revenue"]],
            "opex": g[g["Row Number"] == ROW_MAP["opex"]],
            "rate_base_primary": g[g["Row Number"] == ROW_MAP["rate_base_primary"]],
            "rate_base_fallback": g[g["Row Number"] == ROW_MAP["rate_base_fallback"]],
            "rate_of_return_pct": g[g["Row Number"] == ROW_MAP["rate_of_return_pct"]],
            "total_cost_of_service": g[g["Row Number"] == ROW_MAP["total_cost_of_service"]],
            "throughput_barrels": g[g["Row Number"] == ROW_MAP["throughput_barrels"]],
        }

        for yc in year_cols:
            year = int(yc.replace("Q4", ""))

            rate_base_val = v(sel["rate_base_primary"], yc)
            if pd.isna(rate_base_val):
                rate_base_val = v(sel["rate_base_fallback"], yc)

            rows.append({
                "company": str(company).strip(),
                "ownership": ownership.strip(),
                "year": year,
                "revenue": v(sel["target_revenue"], yc),
                "opex": v(sel["opex"], yc),
                "rate_base": rate_base_val,
                "rate_of_return_pct": v(sel["rate_of_return_pct"], yc),
                "total_cost_of_service": v(sel["total_cost_of_service"], yc),
                "throughput_barrels": v(sel["throughput_barrels"], yc),
            })

    tidy = pd.DataFrame(rows)
    tidy = tidy.dropna(subset=["revenue"]).reset_index(drop=True)
    return tidy


def train_and_evaluate(tidy_df: pd.DataFrame, random_state: int = 42) -> tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
    features = [
        "opex",
        "rate_base",
        "rate_of_return_pct",
        "total_cost_of_service",
        "throughput_barrels",
    ]

    data = tidy_df[["company", "ownership", "year"] + features + ["revenue"]].copy()
    X, y = data[features], data["revenue"]

    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=random_state)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    meta_test = data.iloc[test_idx][["company", "ownership", "year", "revenue"]].reset_index(drop=True)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    results = meta_test.copy()
    results["predicted_revenue"] = preds
    results["error"] = results["predicted_revenue"] - results["revenue"]
    results["abs_error"] = results["error"].abs()
    valid = results["revenue"] > 0
    results.loc[valid, "pct_error"] = results.loc[valid, "error"] / results.loc[valid, "revenue"] * 100.0
    results.to_csv("revenue_predictions_test.csv", index=False)

    # Optional: save feature importances if available
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).to_csv("feature_importances.csv", index=False)

    print("\n=== Model Performance (test set) ===")
    print(f"R^2: {r2:.3f}  |  RMSE: {rmse:,.0f}  |  MAE: {mae:,.0f}")
    print("Saved: revenue_predictions_test.csv" + (", feature_importances.csv" if hasattr(model, "feature_importances_") else ""))

    # Print Top 10 in the test set by actual revenue
    def fmt_money(x: float) -> str:
        try:
            return f"${x:,.0f}"
        except Exception:
            return ""

    def fmt_pct(x: float) -> str:
        try:
            return f"{x:.1f}%"
        except Exception:
            return ""

    preview = (
        results.sort_values("revenue", ascending=False)
        .head(10)
        .loc[:, ["company", "year", "revenue", "predicted_revenue", "error", "pct_error"]]
        .rename(columns={"revenue": "Actual", "predicted_revenue": "Predicted", "error": "Error ($)", "pct_error": "Error (%)"})
    )
    print("\nTop 10 companies by actual revenue (test set):")
    with pd.option_context("display.max_rows", 12, "display.max_colwidth", 40):
        print(
            preview.to_string(
                index=False,
                formatters={
                    "Actual": fmt_money,
                    "Predicted": fmt_money,
                    "Error ($)": fmt_money,
                    "Error (%)": lambda v: "" if pd.isna(v) else fmt_pct(v),
                },
            )
        )

    return pipeline, results, {"r2": r2, "rmse": rmse, "mae": mae}


def simple_company_cagr_map(history: pd.DataFrame, max_years: int = 5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for company, g in history[["company", "year", "revenue"]].dropna().groupby("company"):
        g = g[g["revenue"] > 0].sort_values("year")
        if len(g) < 2:
            continue
        y_end = int(g["year"].max())
        y_start = max(int(g["year"].min()), y_end - max_years + 1)
        g = g[(g["year"] >= y_start) & (g["year"] <= y_end)]
        if len(g) < 2:
            continue
        rev_start = float(g.iloc[0]["revenue"]) or 1.0
        rev_end = float(g.iloc[-1]["revenue"]) or rev_start
        years = int(g.iloc[-1]["year"]) - int(g.iloc[0]["year"]) or 1
        cagr = (rev_end / rev_start) ** (1.0 / years) - 1.0
        out[company] = float(np.clip(cagr, -0.3, 0.3))
    out["__median__"] = float(np.clip(np.median(list(out.values())) if out else 0.0, -0.1, 0.1))
    return out


def forecast_2025_2027(tidy_df: pd.DataFrame, pipeline: Pipeline, years: List[int]) -> pd.DataFrame:
    features = [
        "opex",
        "rate_base",
        "rate_of_return_pct",
        "total_cost_of_service",
        "throughput_barrels",
    ]

    latest = (
        tidy_df.sort_values(["company", "year"], ascending=[True, False])
        .groupby("company", as_index=False)
        .nth(0)
        .reset_index(drop=True)
    )

    base = latest[["company", "ownership"] + features].copy()

    future = []
    for y in years:
        f = base.copy()
        f["year"] = int(y)
        future.append(f)
    future_df = pd.concat(future, ignore_index=True)

    preds = pipeline.predict(future_df[features])
    future_df["predicted_revenue"] = preds

    cagr_map = simple_company_cagr_map(tidy_df)
    def growth(company: str, year: int) -> float:
        cagr = cagr_map.get(company, cagr_map.get("__median__", 0.0))
        horizon = max(0, int(year) - 2024)
        return float((1.0 + cagr) ** horizon)

    future_df["predicted_revenue"] = [
        float(p) * growth(c, y) for p, c, y in zip(future_df["predicted_revenue"], future_df["company"], future_df["year"])
    ]

    out = future_df[["company", "ownership", "year", "predicted_revenue"]].copy()
    out["revenue"] = np.nan
    out["error"] = np.nan
    out["abs_error"] = np.nan
    out["pct_error"] = np.nan
    out = out[["company", "ownership", "year", "revenue", "predicted_revenue", "error", "abs_error", "pct_error"]]

    out.to_csv("revenue_forecasts_2025_2027.csv", index=False)
    out.to_csv("revenue_forecasts_2025_2027_cagr.csv", index=False)

    top_frames = []
    for y in years:
        top_y = (
            out[out["year"] == y]
            .sort_values("predicted_revenue", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top_frames.append(top_y)
    top10 = pd.concat(top_frames, ignore_index=True)
    top10.to_csv("top10_revenue_2025_2027.csv", index=False)
    top10.to_csv("top10_revenue_sum_2025_2027_cagr.csv", index=False)

    print("Saved: revenue_forecasts_2025_2027.csv, top10_revenue_2025_2027.csv")
    # Print Top 10 per forecast year by predicted revenue
    def fmt_money(x: float) -> str:
        try:
            return f"${x:,.0f}"
        except Exception:
            return ""

    print("\n=== Top 10 Predicted Revenues by Year (Forecast) ===")
    for y in years:
        top_y = (
            out[out["year"] == y]
            .sort_values("predicted_revenue", ascending=False)
            .head(10)
            .loc[:, ["company", "year", "predicted_revenue"]]
            .rename(columns={"predicted_revenue": "Predicted"})
        )
        print(f"\nYear {y}:")
        with pd.option_context("display.max_rows", 12, "display.max_colwidth", 40):
            print(top_y.to_string(index=False, formatters={"Predicted": fmt_money}))
    return out


if __name__ == "__main__":
    # 1) Load and tidy
    csv_path = "EastDaley_Form6_pg700_2025_06(Form 6 Page 700).csv"
    raw = load_csv(csv_path)
    tidy = tidy_dataset(raw)

    # 2) Train + evaluate, save test predictions
    model, test_predictions, metrics = train_and_evaluate(tidy)

    # 3) Forecast 2025–2027, save full results and top-10 per year
    forecast_2025_2027(tidy, model, [2025, 2026, 2027])