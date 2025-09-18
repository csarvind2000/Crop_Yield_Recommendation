#!/usr/bin/env python3
"""
Manuscript 1 (Advanced): Robust Crop Yield Modeling
Features:
- Robust preprocessing (impute/scale numerics, impute/one-hot categoricals)
- Models: RandomForest, HistGradientBoosting, HuberRegressor (cost-insensitive)
- Prediction intervals via Quantile GradientBoosting
- Log-target modeling (log1p) with back-transform to original units for metrics
- Winsorization (clipping) of target outliers by quantiles
- Evaluation modes:
    * random split (default 80/20)
    * LOYO  = Leave-One-Year-Out (temporal generalization)
    * LORO  = Leave-One-Region-Out (spatial generalization)
- Permutation importance (original feature names)
- Reproducible outputs (CSVs + PNGs) per evaluation mode

Usage examples:
  python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_random
  python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_loyo --eval loyo
  python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_loro --eval loro
Common flags:
  --target auto           # auto-detect a column containing 'yield' (case-insensitive)
  --log_target            # enable log1p target modeling (default: enabled)
  --no_log_target         # disable log1p target modeling
  --winsor 0.01           # clip target to [q, 1-q] (default: 0.01). Use 0 to disable.
  --year_col YEAR         # override year column name
  --region_col REGION     # override region column name
"""

import argparse, re, os, json, math, warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Utilities --------------------
def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def find_target_column(df: pd.DataFrame, target_arg: str) -> Optional[str]:
    if target_arg and target_arg != "auto":
        if target_arg in df.columns: return target_arg
        raise ValueError(f"Specified target column '{target_arg}' not found.")
    patt = re.compile(r"yield", re.I)
    cand = [c for c in df.columns if patt.search(c)]
    cand_num = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
    if cand_num: return cand_num[0]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[-1] if num_cols else None

def detect_year_col(df: pd.DataFrame, override: Optional[str]=None) -> Optional[str]:
    if override and override in df.columns:
        return override
    # exact match first
    for c in df.columns:
        if c.lower() == "year":
            return c
    # heuristic: int columns mostly between 1990 and 2035
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce").dropna()
            if len(s2) >= max(5, int(0.2*len(s))):
                frac_yearlike = ((s2 >= 1990) & (s2 <= 2035)).mean()
                if frac_yearlike > 0.5:
                    return c
    return None

def detect_region_col(df: pd.DataFrame, override: Optional[str]=None) -> Optional[str]:
    if override and override in df.columns:
        return override
    candidates = {"region","state","district","location","site","zone","province","county"}
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None

def build_preprocessor(df: pd.DataFrame, ycol: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if ycol in num_cols:
        num_cols.remove(ycol)
    cat_cols = [c for c in df.columns if c not in num_cols + [ycol]]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])
    return pre, num_cols, cat_cols

def winsorize_series(y: pd.Series, q: float) -> pd.Series:
    if q <= 0:
        return y
    lo, hi = y.quantile(q), y.quantile(1-q)
    return y.clip(lo, hi)

def back_transform(pred: np.ndarray, use_log: bool) -> np.ndarray:
    return np.expm1(pred) if use_log else pred

def forward_transform(y: pd.Series, use_log: bool) -> pd.Series:
    return np.log1p(y) if use_log else y

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def evaluate_and_save(y_true_raw, y_pred_raw, out_dir: Path, prefix: str):
    # y_* are in original units
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r  = r2_score(y_true_raw, y_pred_raw)
    r_ = rmse(y_true_raw, y_pred_raw)
    pd.DataFrame({"metric":["MAE","RMSE","R2"], "value":[mae, r_, r]}).to_csv(out_dir/f"{prefix}_metrics.csv", index=False)
    # plot
    plt.figure()
    plt.scatter(y_true_raw, y_pred_raw, s=12)
    vmin, vmax = float(min(y_true_raw.min(), y_pred_raw.min())), float(max(y_true_raw.max(), y_pred_raw.max()))
    plt.plot([vmin, vmax], [vmin, vmax])
    plt.xlabel("True Yield"); plt.ylabel("Predicted Yield"); plt.title(f"{prefix}: Predicted vs True")
    plt.tight_layout(); plt.savefig(out_dir/f"{prefix}_pred_vs_true.png", dpi=180); plt.close()
    return mae, r_, r

# -------------------- Core training --------------------
def fit_models(pre: ColumnTransformer):
    models = {
        "RandomForest": Pipeline([("pre", pre),
           ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))]),
        "HistGB": Pipeline([("pre", pre),
           ("model", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=42))]),
        # Huber is robust to outliers; requires dense numeric features -> works through the pipeline
        "Huber": Pipeline([("pre", pre),
           ("model", HuberRegressor())])
    }
    # Simple stack (rf + hgb -> hgb) optional; can be added if needed
    return models

def quantile_intervals(pre: ColumnTransformer, X_train, y_train_trans, X_test,
                       use_log: bool, out_dir: Path, prefix: str):
    # Fit quantile models on preprocessed features (in transformed target space)
    Xt_tr = pre.fit_transform(X_train)
    Xt_te = pre.transform(X_test)

    gb_l = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=500, learning_rate=0.06, random_state=42)
    gb_u = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=500, learning_rate=0.06, random_state=42)
    gb_m = GradientBoostingRegressor(loss="squared_error", n_estimators=500, learning_rate=0.06, random_state=42)

    gb_l.fit(Xt_tr, y_train_trans)
    gb_u.fit(Xt_tr, y_train_trans)
    gb_m.fit(Xt_tr, y_train_trans)

    p10 = back_transform(gb_l.predict(Xt_te), use_log)
    p50 = back_transform(gb_m.predict(Xt_te), use_log)
    p90 = back_transform(gb_u.predict(Xt_te), use_log)

    pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}).to_csv(out_dir/f"{prefix}_intervals.csv", index=False)
    return p10, p50, p90

def run_random_split(df: pd.DataFrame, ycol: str, pre: ColumnTransformer, use_log: bool, winsor_q: float, out_dir: Path):
    X = df.drop(columns=[ycol])
    y_raw = df[ycol].astype(float)
    y_raw = winsorize_series(y_raw, winsor_q)
    y_trans = forward_transform(y_raw, use_log)

    X_train, X_test, y_train_trans, y_test_trans, y_train_raw, y_test_raw = train_test_split(
        X, y_trans, y_raw, test_size=0.2, random_state=42
    )

    models = fit_models(pre)
    rows = []
    preds_collect = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train_trans)
        pred_tr = back_transform(pipe.predict(X_train), use_log)
        pred_te = back_transform(pipe.predict(X_test), use_log)
        mae = mean_absolute_error(y_test_raw, pred_te)
        rows.append({"model": name,
                     "train_MAE": mean_absolute_error(y_train_raw, pred_tr),
                     "train_RMSE": rmse(y_train_raw, pred_tr),
                     "train_R2": r2_score(y_train_raw, pred_tr),
                     "test_MAE": mae,
                     "test_RMSE": rmse(y_test_raw, pred_te),
                     "test_R2": r2_score(y_test_raw, pred_te)})
        preds_collect[name] = pred_te

    perf = pd.DataFrame(rows).sort_values("test_MAE")
    perf.to_csv(out_dir/"random_model_performance.csv", index=False)

    # Intervals (independent models)
    p10, p50, p90 = quantile_intervals(pre, X_train, y_train_trans, X_test, use_log, out_dir, "random")
    covered = ((y_test_raw.values >= p10) & (y_test_raw.values <= p90)).mean()
    pd.DataFrame({"empirical_80pct_coverage":[covered]}).to_csv(out_dir/"random_interval_coverage.csv", index=False)

    # Permutation importance on best
    best_name = perf.iloc[0]["model"]
    best_pipe = models[best_name]
    best_pipe.fit(X_train, y_train_trans)
    r = permutation_importance(best_pipe, X_test, y_test_trans, n_repeats=5, random_state=42, n_jobs=-1,
                               scoring="neg_mean_absolute_error")
    pd.DataFrame({"feature": list(X_test.columns), "importance": r.importances_mean})\
        .sort_values("importance", ascending=False).head(30)\
        .to_csv(out_dir/"random_permutation_importance_top30.csv", index=False)

    # Plots
    plt.figure(); plt.bar(perf["model"], perf["test_MAE"]); plt.ylabel("Test MAE"); plt.title("Random Split — Test MAE"); plt.tight_layout()
    plt.savefig(out_dir/"random_model_comparison_mae.png", dpi=180); plt.close()

    pred_best = back_transform(best_pipe.predict(X_test), use_log)
    plt.figure(); plt.scatter(y_test_raw, pred_best, s=12)
    vmin, vmax = float(min(y_test_raw.min(), pred_best.min())), float(max(y_test_raw.max(), pred_best.max()))
    plt.plot([vmin, vmax], [vmin, vmax])
    plt.xlabel("True Yield"); plt.ylabel("Predicted Yield"); plt.title(f"Random Split — Predicted vs True ({best_name})")
    plt.tight_layout(); plt.savefig(out_dir/"random_pred_vs_true_best.png", dpi=180); plt.close()

def run_loyo(df: pd.DataFrame, ycol: str, year_col: str, pre: ColumnTransformer, use_log: bool, winsor_q: float, out_dir: Path):
    ensure_dir(out_dir)
    years = sorted([x for x in df[year_col].dropna().unique()])
    rows = []
    all_true, all_pred = [], []
    for yr in years:
        tr = df[df[year_col] != yr].copy()
        te = df[df[year_col] == yr].copy()
        if len(te) == 0 or len(tr) == 0: 
            continue
        X_tr, X_te = tr.drop(columns=[ycol]), te.drop(columns=[ycol])
        y_tr_raw = winsorize_series(tr[ycol].astype(float), winsor_q)
        y_te_raw = winsorize_series(te[ycol].astype(float), winsor_q)
        y_tr = forward_transform(y_tr_raw, use_log)
        y_te = forward_transform(y_te_raw, use_log)

        models = fit_models(pre)
        fold_perf = []
        best_mae, best_name, best_pred = 1e18, None, None
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            pred_te = back_transform(pipe.predict(X_te), use_log)
            mae = mean_absolute_error(y_te_raw, pred_te)
            fold_perf.append((name, mae, rmse(y_te_raw, pred_te), r2_score(y_te_raw, pred_te)))
            if mae < best_mae:
                best_mae, best_name, best_pred = mae, name, pred_te

        for name, mae, rm, r2 in fold_perf:
            rows.append({"year": yr, "model": name, "MAE": mae, "RMSE": rm, "R2": r2})
        # keep best predictions across folds for global plot
        all_true.append(y_te_raw.values)
        all_pred.append(best_pred)

    perf = pd.DataFrame(rows)
    perf.to_csv(out_dir/"loyo_fold_metrics.csv", index=False)
    # aggregate by model
    agg = perf.groupby("model")[["MAE","RMSE","R2"]].mean().reset_index()
    agg.to_csv(out_dir/"loyo_model_avg_metrics.csv", index=False)

    # Global plot using concatenated best predictions
    if all_true and all_pred:
        y_true_all = np.concatenate(all_true)
        y_pred_all = np.concatenate(all_pred)
        evaluate_and_save(y_true_all, y_pred_all, out_dir, "loyo_overall")

def run_loro(df: pd.DataFrame, ycol: str, region_col: str, pre: ColumnTransformer, use_log: bool, winsor_q: float, out_dir: Path):
    ensure_dir(out_dir)
    regs = sorted([str(x) for x in df[region_col].dropna().unique()])
    rows = []
    all_true, all_pred = [], []
    for rg in regs:
        tr = df[df[region_col].astype(str) != rg].copy()
        te = df[df[region_col].astype(str) == rg].copy()
        if len(te) == 0 or len(tr) == 0: 
            continue
        X_tr, X_te = tr.drop(columns=[ycol]), te.drop(columns=[ycol])
        y_tr_raw = winsorize_series(tr[ycol].astype(float), winsor_q)
        y_te_raw = winsorize_series(te[ycol].astype(float), winsor_q)
        y_tr = forward_transform(y_tr_raw, use_log)
        y_te = forward_transform(y_te_raw, use_log)

        models = fit_models(pre)
        fold_perf = []
        best_mae, best_name, best_pred = 1e18, None, None
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            pred_te = back_transform(pipe.predict(X_te), use_log)
            mae = mean_absolute_error(y_te_raw, pred_te)
            fold_perf.append((name, mae, rmse(y_te_raw, pred_te), r2_score(y_te_raw, pred_te)))
            if mae < best_mae:
                best_mae, best_name, best_pred = mae, name, pred_te

        for name, mae, rm, r2 in fold_perf:
            rows.append({"region": rg, "model": name, "MAE": mae, "RMSE": rm, "R2": r2})
        all_true.append(y_te_raw.values)
        all_pred.append(best_pred)

    perf = pd.DataFrame(rows)
    perf.to_csv(out_dir/"loro_fold_metrics.csv", index=False)
    agg = perf.groupby("model")[["MAE","RMSE","R2"]].mean().reset_index()
    agg.to_csv(out_dir/"loro_model_avg_metrics.csv", index=False)

    if all_true and all_pred:
        y_true_all = np.concatenate(all_true)
        y_pred_all = np.concatenate(all_pred)
        evaluate_and_save(y_true_all, y_pred_all, out_dir, "loro_overall")

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--target", default="auto", help="Target column name or 'auto'")
    ap.add_argument("--eval", default="random", choices=["random","loyo","loro"], help="Evaluation mode")
    ap.add_argument("--year_col", default=None, help="Year column name override")
    ap.add_argument("--region_col", default=None, help="Region column name override")
    ap.add_argument("--winsor", type=float, default=0.01, help="Winsorization quantile (e.g., 0.01). Use 0 to disable")
    ap.add_argument("--log_target", dest="log_target", action="store_true", help="Enable log1p target modeling")
    ap.add_argument("--no_log_target", dest="log_target", action="store_false", help="Disable log1p target modeling")
    ap.set_defaults(log_target=True)

    args = ap.parse_args()

    out_dir = Path(args.out); ensure_dir(out_dir)
    df = pd.read_csv(args.data)
    ycol = find_target_column(df, args.target)
    if ycol is None:
        raise RuntimeError("No numeric target column found (tried to detect 'yield').")

    year_col = detect_year_col(df, args.year_col)
    region_col = detect_region_col(df, args.region_col)

    # Build preprocessor on full df schema
    pre, num_cols, cat_cols = build_preprocessor(df, ycol)

    # Save metadata
    meta = {
        "target_column": ycol,
        "year_column": year_col,
        "region_column": region_col,
        "n_rows": int(len(df)),
        "winsor": args.winsor,
        "log_target": bool(args.log_target),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "eval_mode": args.eval
    }
    with open(out_dir/"meta.json","w") as f: json.dump(meta, f, indent=2)

    # Run selected evaluation
    if args.eval == "random":
        run_random_split(df, ycol, pre, args.log_target, args.winsor, out_dir)
    elif args.eval == "loyo":
        if year_col is None:
            raise RuntimeError("Could not detect a 'year' column for LOYO. Use --year_col to set it.")
        run_loyo(df, ycol, year_col, pre, args.log_target, args.winsor, out_dir)
    else:  # loro
        if region_col is None:
            raise RuntimeError("Could not detect a region-like column for LORO. Use --region_col to set it.")
        run_loro(df, ycol, region_col, pre, args.log_target, args.winsor, out_dir)

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()



#python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_random

# If auto-detect fails:
# python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_loyo --eval loyo --year_col Year


#python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_loro --eval loro


# If auto-detect fails:
# python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_loro --eval loro --region_col Region


# Turn off log1p:
#python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_random_nolog --no_log_target
# Use 2% winsorization:
#python m1_pipeline_advanced.py --data Crop_Yield_Prediction.csv --out m1_adv_random_w2 --winsor 0.02

