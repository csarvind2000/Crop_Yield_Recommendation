#!/usr/bin/env python3
"""
Manuscript 2: Crop classification (target = Crop), no Yield column used.

Models compared (macro-F1 selection):
- Logistic Regression (balanced)
- RandomForest
- HistGradientBoosting
- MLP (PyTorch)
- LSTM (features as short sequence)
- Transformer (tabular attention over features)

Artifacts:
- model_comparison.csv
- classification_report_<model>.csv
- confusion_matrix_<model>.png
- best_model.json
- For classical: best_model.joblib (+ label_encoder.joblib, classes.json, scaler.json if needed)
- For DL: best_model.pt (+ label_encoder.joblib, classes.json, scaler.json, model_meta.json)

Inference:
- CSV or single row via CLI flags
- Optional Ollama (Llama 3) tips: --ollama_model llama3 (requires local Ollama)

Usage:
  Train:
    python m2_crop_classifier_dl.py train --data Crop_Yield_Prediction.csv --out m2_outputs

  Infer (single row):
    python m2_crop_classifier_dl.py infer --model_dir m2_outputs \
      --nitrogen 80 --phosphorus 40 --potassium 40 --temperature 26 \
      --humidity 75 --ph 6.5 --rainfall 120 --ollama_model ""

  Infer (CSV):
    python m2_crop_classifier_dl.py infer --model_dir m2_outputs --input_csv new_samples.csv
"""

import argparse, json, math, warnings, os
from pathlib import Path
from typing import Dict, Any, List, Optional
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib

# Optional for Ollama
import requests

# Optional DL (PyTorch)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False

RANDOM_STATE = 42
NUMERIC_FEATURES = ["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH_Value","Rainfall"]
TARGET_COL = "Crop"

# ---------------------- Utilities ----------------------
def require_cols(df: pd.DataFrame, cols: List[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES)
    ])

def save_confusion_matrix(y_true, y_pred, labels, title, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)

def train_val_split(dfX, y, test_size=0.2):
    return train_test_split(dfX, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

# ---------------------- Classical models ----------------------
def train_classical_models(X_train, y_train):
    models = {
        "LogReg": Pipeline([("pre", make_preprocessor()),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "HistGB": Pipeline([("pre", make_preprocessor()),
                            ("clf", HistGradientBoostingClassifier(max_depth=None, max_iter=500,
                                                                   learning_rate=0.06, random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline([("pre", make_preprocessor()),
                                  ("clf", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))])
    }
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
    return models

# ---------------------- DL: helpers ----------------------
def compute_standardizer(X_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in NUMERIC_FEATURES:
        col = pd.to_numeric(X_train[c], errors="coerce").astype(float)
        m = float(np.nanmean(col)); s = float(np.nanstd(col) if np.nanstd(col)>1e-8 else 1.0)
        stats[c] = {"mean": m, "std": s}
    return stats

def apply_standardizer(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    arr = []
    for c in NUMERIC_FEATURES:
        x = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(stats[c]["mean"]).values
        arr.append((x - stats[c]["mean"]) / stats[c]["std"])
    return np.vstack(arr).T  # shape (N, F)

# ---------------------- DL: models ----------------------
class MLPNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class LSTMNet(nn.Module):
    """
    Treat features as a short sequence of length F with 1 channel.
    We first lift scalar to dim=32, run LSTM, take last hidden to classify.
    """
    def __init__(self, in_dim: int, n_classes: int, hid=64):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hid, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hid, n_classes)
    def forward(self, x):
        # x: (B, F)
        b, f = x.shape
        x = x.view(b, f, 1)
        x = self.embed(x)            # (B,F,32)
        out, (h, c) = self.lstm(x)   # h: (1,B,hid)
        hlast = h[-1]                # (B,hid)
        return self.fc(hlast)

class TransformerTab(nn.Module):
    """
    Simple tabular Transformer: each feature is a token.
    We project scalar -> d_model, add learnable feature embeddings, run encoder, pool.
    """
    def __init__(self, in_dim: int, n_classes: int, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.feat_embed = nn.Parameter(torch.randn(in_dim, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, F)
        b, f = x.shape
        x = x.view(b, f, 1)                      # (B,F,1)
        v = self.value_proj(x)                   # (B,F,d)
        # add feature id embedding
        e = self.feat_embed.unsqueeze(0).expand(b, -1, -1)  # (B,F,d)
        h = self.encoder(v + e)                  # (B,F,d)
        h = self.norm(h.mean(dim=1))             # mean-pool tokens
        return self.cls(h)

# ---------------------- DL: training ----------------------
def train_torch_model(model, Xtr, ytr, Xte, yte, epochs=80, lr=1e-3, batch=64, patience=10, device="cpu"):
    torch.manual_seed(RANDOM_STATE)
    model.to(device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    tr_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=batch, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            logits = model(Xte_t.to(device))
            pred = logits.argmax(1).cpu().numpy()
            f1 = f1_score(yte, pred, average="macro")
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device))
        pred = logits.argmax(1).cpu().numpy()
        acc = accuracy_score(yte, pred); macro_f1 = f1_score(yte, pred, average="macro")
    return model, pred, acc, macro_f1

# ---------------------- Ollama helper ----------------------
def make_factline(row: Dict[str, Any]) -> str:
    return (f"N={row['Nitrogen']} P={row['Phosphorus']} K={row['Potassium']}, "
            f"pH={row['pH_Value']}, Temp={row['Temperature']}°C, "
            f"Humidity={row['Humidity']}%, Rainfall={row['Rainfall']} mm")

def ask_ollama(model: str, crop_pred: str, row: Dict[str, Any], timeout=60) -> Optional[str]:
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content":
                 "You are an agronomy assistant. Based on N,P,K, pH, temperature, humidity and rainfall, "
                 "suggest safe, actionable recommendations (5–7 bullets) to improve yield for the given crop."},
                {"role": "user", "content":
                 f"Predicted crop: {crop_pred}\nProfile: {make_factline(row)}\n"
                 "- Recommend N-P-K rates & timing, pH correction if needed, irrigation tips, and any crop-specific best practices."}
            ],
            "stream": False
        }
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
        if r.status_code == 200:
            msg = r.json().get("message", {}).get("content", "")
            return msg.strip() or None
        return f"(Ollama status {r.status_code}: {r.text[:200]})"
    except Exception as e:
        return f"(Ollama call failed: {e})"

# ---------------------- Training command ----------------------
def cmd_train(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    # Drop Yield if present
    if "Yield" in df.columns: df = df.drop(columns=["Yield"])
    require_cols(df, NUMERIC_FEATURES + [TARGET_COL])

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].astype(str))
    X = df[NUMERIC_FEATURES].copy()

    # Save label mapping
    joblib.dump(le, out / "label_encoder.joblib")
    json.dump(le.classes_.tolist(), open(out / "classes.json", "w"), indent=2)

    # Split
    Xtr, Xte, ytr, yte = train_val_split(X, y)

    results = []
    # --- Classical
    classical = train_classical_models(Xtr, ytr)
    for name, pipe in classical.items():
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        macro_f1 = f1_score(yte, pred, average="macro")
        results.append((name, acc, macro_f1, pred, pipe))

        # reports
        rep = classification_report(yte, pred, target_names=le.classes_, output_dict=True, zero_division=0)
        pd.DataFrame(rep).transpose().to_csv(out / f"classification_report_{name}.csv")
        save_confusion_matrix(yte, pred, labels=list(range(len(le.classes_))),
                              title=f"Confusion Matrix — {name}",
                              out_path=out / f"confusion_matrix_{name}.png")

    # --- DL (if torch available)
    scaler_stats = compute_standardizer(Xtr)
    json.dump(scaler_stats, open(out / "scaler.json", "w"), indent=2)
    Xtr_std = apply_standardizer(Xtr, scaler_stats)
    Xte_std = apply_standardizer(Xte, scaler_stats)

    if TORCH_OK:
        device = "cpu"
        n_classes = len(le.classes_)
        in_dim = len(NUMERIC_FEATURES)

        # MLP
        mlp = MLPNet(in_dim, n_classes)
        mlp, pred, acc, macro_f1 = train_torch_model(mlp, Xtr_std, ytr, Xte_std, yte,
                                                     epochs=120, lr=1e-3, batch=64, patience=12, device=device)
        results.append(("MLP", acc, macro_f1, pred, mlp))

        # LSTM
        lstm = LSTMNet(in_dim, n_classes, hid=64)
        lstm, pred, acc, macro_f1 = train_torch_model(lstm, Xtr_std, ytr, Xte_std, yte,
                                                      epochs=140, lr=1e-3, batch=64, patience=15, device=device)
        results.append(("LSTM", acc, macro_f1, pred, lstm))

        # Transformer
        trans = TransformerTab(in_dim, n_classes, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1)
        trans, pred, acc, macro_f1 = train_torch_model(trans, Xtr_std, ytr, Xte_std, yte,
                                                       epochs=160, lr=1e-3, batch=64, patience=15, device=device)
        results.append(("Transformer", acc, macro_f1, pred, trans))

        # Save DL confusion matrices and reports
        for name, acc, macro_f1, pred, model in [r for r in results if r[0] in {"MLP","LSTM","Transformer"}]:
            rep = classification_report(yte, pred, target_names=le.classes_, output_dict=True, zero_division=0)
            pd.DataFrame(rep).transpose().to_csv(out / f"classification_report_{name}.csv")
            save_confusion_matrix(yte, pred, labels=list(range(len(le.classes_))),
                                  title=f"Confusion Matrix — {name}",
                                  out_path=out / f"confusion_matrix_{name}.png")
    else:
        # Note in a file so you remember
        (out / "torch_unavailable.txt").write_text("PyTorch not installed; DL models were skipped.\n"
                                                   "Install with: pip install torch")

    # --- Compare & pick best (macro-F1)
    comp = pd.DataFrame([{"model": m, "accuracy": acc, "macro_f1": mf1} for (m, acc, mf1, _, _) in results])
    comp.sort_values("macro_f1", ascending=False).to_csv(out / "model_comparison.csv", index=False)

    # Best
    best_idx = int(np.argmax([mf1 for (_, _, mf1, _, _) in results]))
    best_name, best_acc, best_f1, best_pred, best_obj = results[best_idx]

    # Persist best
    meta = {
        "best_model": best_name,
        "metric": "macro_f1",
        "macro_f1": float(best_f1),
        "accuracy": float(best_acc),
        "features": NUMERIC_FEATURES,
        "target": TARGET_COL,
        "random_state": RANDOM_STATE
    }
    json.dump(meta, open(out / "best_model.json", "w"), indent=2)

    if best_name in {"LogReg","HistGB","RandomForest"}:
        joblib.dump(best_obj, out / "best_model.joblib")
        (out / "model_type.txt").write_text("sklearn")
    else:
        # DL: save .pt and meta
        if not TORCH_OK:
            raise RuntimeError("Internal: torch not available when saving DL model.")
        import torch
        torch.save(best_obj.state_dict(), out / "best_model.pt")
        json.dump({"dl_model": best_name}, open(out / "model_type.txt", "w"))
        # save architecture meta for reload
        json.dump({"arch": best_name, "in_dim": len(NUMERIC_FEATURES), "n_classes": len(le.classes_)},
                  open(out / "model_meta.json", "w"), indent=2)

    print(f"Best model: {best_name} | macro-F1={best_f1:.3f} | acc={best_acc:.3f}")
    print("Artifacts saved in:", out)

# ---------------------- Inference ----------------------
def load_best(model_dir: Path):
    le = joblib.load(model_dir / "label_encoder.joblib")
    classes = json.load(open(model_dir / "classes.json"))
    model_type = open(model_dir / "model_type.txt").read().strip()
    scaler = json.load(open(model_dir / "scaler.json")) if (model_dir / "scaler.json").exists() else None
    if model_type == "sklearn":
        pipe = joblib.load(model_dir / "best_model.joblib")
        return {"type": "sklearn", "pipe": pipe, "le": le, "classes": classes, "scaler": scaler}
    else:
        if not TORCH_OK:
            raise RuntimeError("This best model is DL, but torch is not installed.")
        meta = json.load(open(model_dir / "model_meta.json"))
        arch = meta["arch"]; in_dim = meta["in_dim"]; n_classes = meta["n_classes"]
        # rebuild model
        if arch == "MLP":
            model = MLPNet(in_dim, n_classes)
        elif arch == "LSTM":
            model = LSTMNet(in_dim, n_classes, hid=64)
        elif arch == "Transformer":
            model = TransformerTab(in_dim, n_classes, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1)
        else:
            raise RuntimeError(f"Unknown DL arch: {arch}")
        import torch
        state = torch.load(model_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(state); model.eval()
        return {"type": "dl", "model": model, "le": le, "classes": classes, "scaler": scaler, "arch": arch}

def predict_proba_best(loaded, Xdf: pd.DataFrame) -> pd.DataFrame:
    if loaded["type"] == "sklearn":
        pipe = loaded["pipe"]
        # some sklearn models may lack predict_proba
        try:
            proba = pipe.predict_proba(Xdf)
        except Exception:
            pred = pipe.predict(Xdf)
            proba = None
    else:
        # DL
        import torch
        stats = loaded["scaler"]
        X_std = apply_standardizer(Xdf, stats)
        x = torch.tensor(X_std, dtype=torch.float32)
        with torch.no_grad():
            logits = loaded["model"](x)
            proba_t = torch.softmax(logits, dim=1)
            proba = proba_t.numpy()
    # assemble outputs
    if loaded["type"] == "sklearn":
        pred_idx = loaded["pipe"].predict(Xdf)
    else:
        pred_idx = np.argmax(proba, axis=1)
    pred_lbl = loaded["le"].inverse_transform(pred_idx)
    out = pd.DataFrame({"PredictedCrop": pred_lbl})
    if proba is not None:
        classes = np.array(loaded["classes"])
        top = np.argsort(-proba, axis=1)
        k = min(3, proba.shape[1])
        for i in range(k):
            idx = top[:, i]
            out[f"Top{i+1}_Class"] = classes[idx]
            out[f"Top{i+1}_Prob"] = np.round(proba[np.arange(len(proba)), idx], 4)
    return out

# ---------------------- CLI inference helpers ----------------------
def make_row_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "Nitrogen": args.nitrogen,
        "Phosphorus": args.phosphorus,
        "Potassium": args.potassium,
        "Temperature": args.temperature,
        "Humidity": args.humidity,
        "pH_Value": args.ph,
        "Rainfall": args.rainfall
    }

def cmd_infer(args):
    model_dir = Path(args.model_dir)
    loaded = load_best(model_dir)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        # ignore Crop if present
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        require_cols(df, NUMERIC_FEATURES)
        Xdf = df[NUMERIC_FEATURES].copy()
        preds = predict_proba_best(loaded, Xdf)
        out = model_dir / "inference_results.csv"
        preds.to_csv(out, index=False)
        print("Saved predictions to:", out)
        # Optional Ollama (recommendations for first few rows)
        if args.ollama_model:
            n = min(5, len(Xdf))
            print(f"Asking Ollama for {n} rows...")
            for i in range(n):
                row = Xdf.iloc[i].to_dict()
                crop = preds.iloc[i]["PredictedCrop"]
                txt = ask_ollama(args.ollama_model, crop, row)
                (model_dir / f"ollama_row{i+1}.txt").write_text(txt or "")
            print("Ollama recommendations saved as ollama_row*.txt")
    else:
        row = make_row_from_args(args)
        Xdf = pd.DataFrame([row], columns=NUMERIC_FEATURES)
        preds = predict_proba_best(loaded, Xdf)
        print(preds.to_string(index=False))
        if args.ollama_model:
            txt = ask_ollama(args.ollama_model, preds.iloc[0]["PredictedCrop"], row)
            print("\n--- Recommendations (Ollama) ---\n", txt or "(none)")

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)

    f = sub.add_parser("infer")
    f.add_argument("--model_dir", required=True)
    f.add_argument("--input_csv", default=None)
    f.add_argument("--nitrogen", type=float, default=None)
    f.add_argument("--phosphorus", type=float, default=None)
    f.add_argument("--potassium", type=float, default=None)
    f.add_argument("--temperature", type=float, default=None)
    f.add_argument("--humidity", type=float, default=None)
    f.add_argument("--ph", type=float, default=None)
    f.add_argument("--rainfall", type=float, default=None)
    f.add_argument("--ollama_model", type=str, default="")

    args = ap.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "infer":
        # if no CSV, ensure single-row args present
        if args.input_csv is None:
            for k in ["nitrogen","phosphorus","potassium","temperature","humidity","ph","rainfall"]:
                if getattr(args, k) is None:
                    raise SystemExit(f"--{k} is required for single-row inference (or provide --input_csv)")
        cmd_infer(args)

if __name__ == "__main__":
    main()
