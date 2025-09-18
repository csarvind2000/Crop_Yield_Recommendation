#!/usr/bin/env python3
"""
Crop-wise Yield Prediction Pipeline (Headless + Fit Line + Units)

- Per-crop models: RandomForest, HistGB, Huber
- Preprocessing per crop: numeric median+scale, categorical most-freq+one-hot (EXCLUDES crop col)
- Target winsorization (default 1%)
- Log-target with smearing bias correction on back-transform
- Evaluation: random (80/20), LOYO (per crop), LORO (per crop)
- Quantile prediction intervals (p10/p50/p90)
- Permutation importance (original input columns)
- Headless plotting (matplotlib Agg) — no Tkinter warnings
- Pred vs True shows ideal 1:1 and best-fit regression line
"""

# ---- headless backend (MUST be before pyplot import) ----
import matplotlib
matplotlib.use("Agg")

import argparse, re, json, math, warnings
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Utils --------------------
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

def detect_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

def detect_year_col(df: pd.DataFrame, override: Optional[str]=None) -> Optional[str]:
    if override and override in df.columns: return override
    for c in df.columns:
        if c.lower() == "year": return c
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce").dropna()
            if len(s2) >= max(5, int(0.2*len(s))):
                frac_yearlike = ((s2 >= 1990) & (s2 <= 2035)).mean()
                if frac_yearlike > 0.5: return c
    return None

def detect_region_col(df: pd.DataFrame, override: Optional[str]=None) -> Optional[str]:
    if override and override in df.columns: return override
    candidates = {"region","state","district","location","site","zone","province","county"}
    for c in df.columns:
        if c.lower() in candidates: return c
    return None

def build_preprocessor(df: pd.DataFrame, ycol: str, crop_col: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if ycol in num_cols: num_cols.remove(ycol)
    cat_cols = [c for c in df.columns if c not in num_cols + [ycol]]
    if crop_col in cat_cols:
        cat_cols.remove(crop_col)  # avoid leakage within a single-crop subset
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])
    return pre, num_cols, cat_cols

def winsorize_series(y: pd.Series, q: float) -> pd.Series:
    if q <= 0: return y
    lo, hi = y.quantile(q), y.quantile(1-q)
    return y.clip(lo, hi)

def forward_transform(y: pd.Series, use_log: bool) -> pd.Series:
    return np.log1p(y) if use_log else y

def smearing_variance(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    resid = y_true_log - y_pred_log
    return float(np.var(resid))

def back_transform(pred: np.ndarray, use_log: bool, sigma2: Optional[float]=None) -> np.ndarray:
    if not use_log: return pred
    if sigma2 is None: return np.expm1(pred)
    return np.expm1(pred + 0.5*sigma2)

def quantile_intervals(pre: ColumnTransformer, X_train, y_train_trans, X_test, use_log: bool, out_dir: Path, prefix: str):
    Xt_tr = pre.fit_transform(X_train)
    Xt_te = pre.transform(X_test)
    gb_l = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=400, learning_rate=0.06, random_state=42)
    gb_u = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=400, learning_rate=0.06, random_state=42)
    gb_m = GradientBoostingRegressor(loss="squared_error", n_estimators=400, learning_rate=0.06, random_state=42)
    gb_l.fit(Xt_tr, y_train_trans); gb_u.fit(Xt_tr, y_train_trans); gb_m.fit(Xt_tr, y_train_trans)
    p10 = back_transform(gb_l.predict(Xt_te), use_log)
    p50 = back_transform(gb_m.predict(Xt_te), use_log)
    p90 = back_transform(gb_u.predict(Xt_te), use_log)
    pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}).to_csv(out_dir/f"{prefix}_intervals.csv", index=False)
    return p10, p50, p90

def plot_pred_vs_true(y_true, y_pred, title, xlabel, ylabel, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure()
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    lo = float(min(y_true.min(), y_pred.min())); hi = float(max(y_true.max(), y_pred.max()))
    # ideal 1:1
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    # best-fit regression (y_pred = a*y_true + b)
    reg = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    y_fit = reg.predict(np.array([lo, hi]).reshape(-1, 1))
    plt.plot([lo, hi], y_fit, linewidth=2)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title}\nFit: y = {reg.coef_[0]:.3f}x + {reg.intercept_:.1f}  |  R² = {r2_score(y_true, y_pred):.3f}")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()

# -------------------- Core per-crop run --------------------
def run_per_crop(df_crop: pd.DataFrame, crop_name: str, ycol: str, crop_col: str,
                 use_log: bool, winsor_q: float, eval_mode: str, out_dir: Path,
                 year_col: Optional[str], region_col: Optional[str], min_rows: int,
                 yield_units: str):

    if len(df_crop) < max(30, min_rows):
        return None  # too small; skip

    pre_local, _, _ = build_preprocessor(df_crop, ycol, crop_col)

    # random split within this crop
    if eval_mode == "random":
        X = df_crop.drop(columns=[ycol])
        y_raw = winsorize_series(df_crop[ycol].astype(float), winsor_q)
        y_trans = forward_transform(y_raw, use_log)
        X_tr, X_te, y_tr_trans, y_te_trans, y_tr_raw, y_te_raw = train_test_split(
            X, y_trans, y_raw, test_size=0.2, random_state=42
        )

        models = {
            "RandomForest": Pipeline([("pre", pre_local),
               ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))]),
            "HistGB": Pipeline([("pre", pre_local),
               ("model", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=42))]),
            "Huber": Pipeline([("pre", pre_local), ("model", HuberRegressor())])
        }

        rows = []
        best_name, best_pred = None, None
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr_trans)
            pred_te_log = pipe.predict(X_te)
            sigma2 = smearing_variance(y_te_trans.values, pred_te_log) if use_log else None
            pred_te = back_transform(pred_te_log, use_log, sigma2)
            tr_pred = back_transform(pipe.predict(X_tr), use_log, sigma2)
            rows.append({
                "model": name,
                f"train_MAE ({yield_units})": mean_absolute_error(y_tr_raw, tr_pred),
                f"train_RMSE ({yield_units})": rmse(y_tr_raw, tr_pred),
                "train_R2": r2_score(y_tr_raw, tr_pred),
                f"test_MAE ({yield_units})": mean_absolute_error(y_te_raw, pred_te),
                f"test_RMSE ({yield_units})": rmse(y_te_raw, pred_te),
                "test_R2": r2_score(y_te_raw, pred_te),
            })
            if best_name is None or rows[-1][f"test_MAE ({yield_units})"] < [r for r in rows if r["model"]==best_name][0][f"test_MAE ({yield_units})"]:
                best_name = name; best_pred = pred_te

        perf = pd.DataFrame(rows).sort_values(f"test_MAE ({yield_units})")
        perf.to_csv(out_dir/"random_model_performance.csv", index=False)

        # intervals + coverage
        p10, p50, p90 = quantile_intervals(pre_local, X_tr, y_tr_trans, X_te, use_log, out_dir, "random")
        covered = ((y_te_raw.values >= p10) & (y_te_raw.values <= p90)).mean()
        pd.DataFrame({"empirical_80pct_coverage":[covered]}).to_csv(out_dir/"random_interval_coverage.csv", index=False)

        # permutation importance (original columns)
        best_pipe = models[best_name]; best_pipe.fit(X_tr, y_tr_trans)
        r = permutation_importance(best_pipe, X_te, y_te_trans, n_repeats=5, random_state=42, n_jobs=-1,
                                   scoring="neg_mean_absolute_error")
        pd.DataFrame({"feature": list(X_te.columns), "importance": r.importances_mean})\
          .sort_values("importance", ascending=False).head(30)\
          .to_csv(out_dir/"random_permutation_importance_top30.csv", index=False)

        # plots
        plt.figure(); plt.bar(perf["model"], perf[f"test_MAE ({yield_units})"])
        plt.ylabel(f"Test MAE ({yield_units})"); plt.title("Random Split — Test MAE"); plt.tight_layout()
        plt.savefig(out_dir/"random_model_comparison_mae.png", dpi=180); plt.close()

        plot_pred_vs_true(
            y_true=y_te_raw, y_pred=best_pred,
            title=f"Random Split — Predicted vs True ({best_name})",
            xlabel=f"True Yield ({yield_units})", ylabel=f"Predicted Yield ({yield_units})",
            out_path=out_dir/"random_pred_vs_true_best.png"
        )

        print(f"[{crop_name}] Best: {best_name} | Test MAE: {perf.iloc[0][f'test_MAE ({yield_units})']:.2f} {yield_units}")
        return {"crop": crop_name, "n_rows": len(df_crop), "best_model": best_name,
                "test_MAE": float(perf.iloc[0][f"test_MAE ({yield_units})"]),
                "test_RMSE": float(perf.iloc[0][f"test_RMSE ({yield_units})"]),
                "test_R2": float(perf.iloc[0]["test_R2"]), "coverage80": float(covered)}

    # LOYO (per crop)
    elif eval_mode == "loyo":
        if year_col is None: raise RuntimeError("LOYO requires a year column")
        years = sorted([x for x in df_crop[year_col].dropna().unique()])
        rows = []; all_true=[]; all_pred=[]
        for yr in years:
            tr = df_crop[df_crop[year_col] != yr].copy()
            te = df_crop[df_crop[year_col] == yr].copy()
            if len(te)==0 or len(tr)==0: continue
            X_tr, X_te = tr.drop(columns=[ycol]), te.drop(columns=[ycol])
            y_tr_raw = winsorize_series(tr[ycol].astype(float), winsor_q)
            y_te_raw = winsorize_series(te[ycol].astype(float), winsor_q)
            y_tr = forward_transform(y_tr_raw, use_log)
            y_te = forward_transform(y_te_raw, use_log)

            pre_local, _, _ = build_preprocessor(df_crop, ycol, crop_col)
            models = {
                "RandomForest": Pipeline([("pre", pre_local),
                   ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))]),
                "HistGB": Pipeline([("pre", pre_local),
                   ("model", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=42))]),
                "Huber": Pipeline([("pre", pre_local), ("model", HuberRegressor())])
            }
            best_mae, best_pred = 1e18, None
            for name, pipe in models.items():
                pipe.fit(X_tr, y_tr)
                pred_te_log = pipe.predict(X_te)
                sigma2 = smearing_variance(y_te.values, pred_te_log) if use_log else None
                pred_te = back_transform(pred_te_log, use_log, sigma2)
                mae = mean_absolute_error(y_te_raw, pred_te)
                rows.append({"year": yr, "model": name, f"MAE ({yield_units})": mae,
                             f"RMSE ({yield_units})": rmse(y_te_raw, pred_te), "R2": r2_score(y_te_raw, pred_te)})
                if mae < best_mae:
                    best_mae, best_pred = mae, pred_te
            all_true.append(y_te_raw.values); all_pred.append(best_pred)
        perf = pd.DataFrame(rows); perf.to_csv(out_dir/"loyo_fold_metrics.csv", index=False)
        agg = perf.groupby("model")[[f"MAE ({yield_units})",f"RMSE ({yield_units})","R2"]].mean().reset_index()
        agg.to_csv(out_dir/"loyo_model_avg_metrics.csv", index=False)
        if all_true and all_pred:
            y_true_all = np.concatenate(all_true); y_pred_all = np.concatenate(all_pred)
            plot_pred_vs_true(y_true_all, y_pred_all,
                              title="LOYO — Predicted vs True (best per fold)",
                              xlabel=f"True Yield ({yield_units})", ylabel=f"Predicted Yield ({yield_units})",
                              out_path=out_dir/"loyo_overall_pred_vs_true.png")
        print(f"[{crop_name}] LOYO done.")
        return {"crop": crop_name, "n_rows": len(df_crop), "best_model": "per-fold",
                "test_MAE": float(agg[f"MAE ({yield_units})"].min()),
                "test_RMSE": float(agg[f"RMSE ({yield_units})"].min()),
                "test_R2": float(agg["R2"].max())}

    # LORO (per crop)
    else:
        if region_col is None: raise RuntimeError("LORO requires a region column")
        regs = sorted([str(x) for x in df_crop[region_col].dropna().unique()])
        rows = []; all_true=[]; all_pred=[]
        for rg in regs:
            tr = df_crop[df_crop[region_col].astype(str) != rg].copy()
            te = df_crop[df_crop[region_col].astype(str) == rg].copy()
            if len(te)==0 or len(tr)==0: continue
            X_tr, X_te = tr.drop(columns=[ycol]), te.drop(columns=[ycol])
            y_tr_raw = winsorize_series(tr[ycol].astype(float), winsor_q)
            y_te_raw = winsorize_series(te[ycol].astype(float), winsor_q)
            y_tr = forward_transform(y_tr_raw, use_log)
            y_te = forward_transform(y_te_raw, use_log)

            pre_local, _, _ = build_preprocessor(df_crop, ycol, crop_col)
            models = {
                "RandomForest": Pipeline([("pre", pre_local),
                   ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))]),
                "HistGB": Pipeline([("pre", pre_local),
                   ("model", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, random_state=42))]),
                "Huber": Pipeline([("pre", pre_local), ("model", HuberRegressor())])
            }
            best_mae, best_pred = 1e18, None
            for name, pipe in models.items():
                pipe.fit(X_tr, y_tr)
                pred_te_log = pipe.predict(X_te)
                sigma2 = smearing_variance(y_te.values, pred_te_log) if use_log else None
                pred_te = back_transform(pred_te_log, use_log, sigma2)
                mae = mean_absolute_error(y_te_raw, pred_te)
                rows.append({"region": rg, "model": name, f"MAE ({yield_units})": mae,
                             f"RMSE ({yield_units})": rmse(y_te_raw, pred_te), "R2": r2_score(y_te_raw, pred_te)})
                if mae < best_mae:
                    best_mae, best_pred = mae, pred_te
            all_true.append(y_te_raw.values); all_pred.append(best_pred)
        perf = pd.DataFrame(rows); perf.to_csv(out_dir/"loro_fold_metrics.csv", index=False)
        agg = perf.groupby("model")[[f"MAE ({yield_units})",f"RMSE ({yield_units})","R2"]].mean().reset_index()
        agg.to_csv(out_dir/"loro_model_avg_metrics.csv", index=False)
        if all_true and all_pred:
            y_true_all = np.concatenate(all_true); y_pred_all = np.concatenate(all_pred)
            plot_pred_vs_true(y_true_all, y_pred_all,
                              title="LORO — Predicted vs True (best per fold)",
                              xlabel=f"True Yield ({yield_units})", ylabel=f"Predicted Yield ({yield_units})",
                              out_path=out_dir/"loro_overall_pred_vs_true.png")
        print(f"[{crop_name}] LORO done.")
        return {"crop": crop_name, "n_rows": len(df_crop), "best_model": "per-fold",
                "test_MAE": float(agg[f"MAE ({yield_units})"].min()),
                "test_RMSE": float(agg[f"RMSE ({yield_units})"].min()),
                "test_R2": float(agg["R2"].max())}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--target", default="auto", help="Target column or 'auto'")
    ap.add_argument("--crop_col", default="auto", help="Crop column or 'auto' to detect")
    ap.add_argument("--eval", default="random", choices=["random","loyo","loro"], help="Evaluation mode")
    ap.add_argument("--year_col", default=None, help="Year column (for loyo)")
    ap.add_argument("--region_col", default=None, help="Region column (for loro)")
    ap.add_argument("--winsor", type=float, default=0.01, help="Winsorization quantile (0 disables)")
    # Accept BOTH --yield_units and --units (alias) for compatibility
    ap.add_argument("--yield_units", "--units", dest="yield_units", type=str, default="yield units",
                    help="Units label for yield (e.g., 'kg/ha')")
    ap.add_argument("--log_target", dest="log_target", action="store_true", help="Enable log1p target")
    ap.add_argument("--no_log_target", dest="log_target", action="store_false", help="Disable log1p target")
    ap.add_argument("--min_rows", type=int, default=60, help="Minimum rows per crop to train (default 60)")
    ap.set_defaults(log_target=True)

    args = ap.parse_args()
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    ycol = find_target_column(df, args.target)
    if ycol is None:
        raise RuntimeError("No numeric target column found (tried to detect 'yield').")

    crop_col = args.crop_col if args.crop_col != "auto" else detect_col(df, ["Crop","crop","CROP"])
    if crop_col is None:
        raise RuntimeError("Could not detect crop column. Use --crop_col to specify.")
    year_col = args.year_col if args.year_col else detect_year_col(df)
    region_col = args.region_col if args.region_col else detect_region_col(df)

    crops = [c for c in sorted(df[crop_col].dropna().unique())]
    summary_rows = []
    for crop in crops:
        sub = df[df[crop_col] == crop].copy()
        out_dir = out_root / f"crop_{str(crop).replace(' ','_')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {"crop": str(crop), "n_rows": int(len(sub)), "target_column": ycol, "crop_column": crop_col,
                "year_column": year_col, "region_column": region_col, "winsor": args.winsor,
                "log_target": bool(args.log_target), "eval_mode": args.eval, "yield_units": args.yield_units}
        (out_dir/"meta.json").write_text(json.dumps(meta, indent=2))

        try:
            res = run_per_crop(
                sub, str(crop), ycol, crop_col,
                args.log_target, args.winsor, args.eval, out_dir,
                year_col, region_col, args.min_rows, args.yield_units
            )
            if res:
                summary_rows.append(res)
            else:
                summary_rows.append({"crop": str(crop), "n_rows": int(len(sub)),
                                     "best_model": "skipped (too few rows)",
                                     "test_MAE": np.nan, "test_RMSE": np.nan,
                                     "test_R2": np.nan, "coverage80": np.nan})
        except Exception as e:
            summary_rows.append({"crop": str(crop), "n_rows": int(len(sub)),
                                 "best_model": f"error: {e}",
                                 "test_MAE": np.nan, "test_RMSE": np.nan,
                                 "test_R2": np.nan, "coverage80": np.nan})

    pd.DataFrame(summary_rows).to_csv(out_root/"cropwise_summary.csv", index=False)
    print("Done. Outputs in:", out_root)

if __name__ == "__main__":
    main()


# python m1_pipeline_cropwise.py --data Crop_Yield_Prediction.csv --out m1_cropwise_random --eval random --yield_units "kg/ha" --min_rows 0
