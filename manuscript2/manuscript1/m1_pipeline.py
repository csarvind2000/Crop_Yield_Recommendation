#!/usr/bin/env python3
"""
Manuscript 1: Hybrid Gradient–Tree Ensemble for Crop Yield Prediction
Usage:
  python m1_pipeline.py --data Crop_Yield_Prediction.csv --target auto --out outputs_m1
"""
import argparse, re, os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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

def build_preprocessor(df: pd.DataFrame, ycol: str):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if ycol in num_cols: num_cols.remove(ycol)
    cat_cols = [c for c in df.columns if c not in num_cols + [ycol]]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])
    return pre, num_cols, cat_cols

def get_feature_names(pre: ColumnTransformer) -> list:
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names.extend(list(cols))
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            names.extend(list(ohe.get_feature_names_out(cols)))
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV")
    ap.add_argument("--target", default="auto", help="Target column name or 'auto'")
    ap.add_argument("--out", default="outputs_m1", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    ycol = find_target_column(df, args.target)
    if ycol is None: raise RuntimeError("No numeric target column found (tried to detect 'yield').")

    pre, num_cols, cat_cols = build_preprocessor(df, ycol)
    X = df.drop(columns=[ycol]); y = df[ycol].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": Pipeline([("pre", pre),
           ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))]),
        "HistGB": Pipeline([("pre", pre),
           ("model", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=42))]),
    }
    stack = StackingRegressor(
        estimators=[("rf", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
                   ("hgb", HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, random_state=42))],
        final_estimator=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42),
        passthrough=False, n_jobs=-1)
    models["Stacked_Ensemble"] = Pipeline([("pre", pre), ("model", stack)])

    results = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        yp_tr = pipe.predict(X_train)
        yp_te = pipe.predict(X_test)
        results.append({
            "model": name,
            "train_MAE": mean_absolute_error(y_train, yp_tr),
            "train_RMSE": rmse(y_train, yp_tr),
            "train_R2": r2_score(y_train, yp_tr),
            "test_MAE": mean_absolute_error(y_test, yp_te),
            "test_RMSE": rmse(y_test, yp_te),
            "test_R2": r2_score(y_test, yp_te),
        })
    res_df = pd.DataFrame(results).sort_values("test_MAE")
    res_df.to_csv(out_dir/"model_performance.csv", index=False)

    # Prediction intervals via quantile GBDT (using pre.transform)
    X_train_trans = pre.fit_transform(X_train)
    X_test_trans  = pre.transform(X_test)

    gb_lower = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=400, learning_rate=0.06, random_state=42)
    gb_upper = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=400, learning_rate=0.06, random_state=42)
    gb_mean  = GradientBoostingRegressor(loss="squared_error", n_estimators=400, learning_rate=0.06, random_state=42)

    gb_lower.fit(X_train_trans, y_train)
    gb_upper.fit(X_train_trans, y_train)
    gb_mean.fit(X_train_trans, y_train)

    pred_lower = gb_lower.predict(X_test_trans)
    pred_upper = gb_upper.predict(X_test_trans)
    pred_mean  = gb_mean.predict(X_test_trans)

    interval_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": pred_mean,
        "p10": pred_lower,
        "p90": pred_upper
    })
    interval_df["covered_80"] = ((interval_df["y_true"] >= interval_df["p10"]) &
                                 (interval_df["y_true"] <= interval_df["p90"])).astype(int)
    interval_df.to_csv(out_dir/"predictions_with_intervals.csv", index=False)

    coverage = float(interval_df["covered_80"].mean())

    # Permutation importance for best model
    best_name = res_df.iloc[0]["model"]
    best_pipe = models[best_name]; best_pipe.fit(X_train, y_train)
    r = permutation_importance(best_pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1,
                               scoring="neg_mean_absolute_error")
    feat_names = get_feature_names(best_pipe.named_steps["pre"])
    imp = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean}).sort_values("importance", ascending=False).head(30)
    imp.to_csv(out_dir/"permutation_importance_top30.csv", index=False)

    # Plots (matplotlib, no styles/colors)
    plt.figure()
    plt.bar(res_df["model"], res_df["test_MAE"])
    plt.ylabel("Test MAE"); plt.title("Model Comparison (Test MAE)"); plt.tight_layout()
    plt.savefig(out_dir/"model_comparison_mae.png", dpi=180); plt.close()

    yp_best = best_pipe.predict(X_test)
    plt.figure()
    plt.scatter(y_test, yp_best, s=12)
    minv, maxv = float(min(y_test.min(), yp_best.min())), float(max(y_test.max(), yp_best.max()))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("True Yield"); plt.ylabel("Predicted Yield"); plt.title(f"Predicted vs True — {best_name}")
    plt.tight_layout(); plt.savefig(out_dir/"pred_vs_true_best.png", dpi=180); plt.close()

    plt.figure()
    idx = np.arange(len(interval_df))
    plt.plot(idx, interval_df["y_true"].values)
    plt.plot(idx, interval_df["y_pred"].values)
    plt.fill_between(idx, interval_df["p10"].values, interval_df["p90"].values, alpha=0.2)
    plt.xlabel("Test Sample Index"); plt.ylabel("Yield")
    plt.title(f"Prediction Intervals (empirical 80% coverage={coverage:.2f})")
    plt.tight_layout(); plt.savefig(out_dir/"prediction_intervals.png", dpi=180); plt.close()

    # Save meta
    meta = {
        "target_column": ycol,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "numeric_cols": num_cols, "categorical_cols": cat_cols,
        "coverage_80": coverage, "best_model": best_name
    }
    with open(out_dir/"meta.json","w") as f: json.dump(meta, f, indent=2)

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()


#python m1_pipeline.py --data Crop_Yield_Prediction.csv --target auto --out m1_outputs
