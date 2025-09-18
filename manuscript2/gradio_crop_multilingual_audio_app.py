#!/usr/bin/env python3
"""
Multilingual crop prediction + LLM recos with robust two-pass (EN->target) translation
+ Text-to-Speech (gTTS) for spoken recommendations in the selected language.

Run:
  python gradio_crop_multilingual_app.py \
      --model_dir m2_outputs \
      --images_dir images \
      --ollama_model llama3 \
      --port 7860
"""

import argparse, json, re, unicodedata, tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
from PIL import Image, ImageDraw
import gradio as gr

# -------- Optional deps --------
# DL backbone (only if your best model is DL)
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# gTTS for multilingual TTS
try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

NUMERIC_FEATURES = ["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH_Value","Rainfall"]

# -------------------- DL Models --------------------
class MLPNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

class LSTMNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hid=64):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.lstm  = nn.LSTM(input_size=32, hidden_size=hid, num_layers=1, batch_first=True)
        self.fc    = nn.Linear(hid, n_classes)
    def forward(self, x):
        b, f = x.shape
        x = x.view(b, f, 1)
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])

class TransformerTab(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.feat_embed = nn.Parameter(torch.randn(in_dim, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.cls  = nn.Linear(d_model, n_classes)
    def forward(self, x):
        b, f = x.shape
        x = x.view(b, f, 1)
        v = self.value_proj(x)
        e = self.feat_embed.unsqueeze(0).expand(b, -1, -1)
        h = self.encoder(v + e)
        h = self.norm(h.mean(dim=1))
        return self.cls(h)

# -------------------- Artifacts --------------------
def apply_standardizer(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    arr = []
    for c in NUMERIC_FEATURES:
        val = pd.to_numeric(df[c], errors="coerce").astype(float)
        m = stats[c]["mean"]; s = stats[c]["std"] if stats[c]["std"] != 0 else 1.0
        arr.append((val.fillna(m).values - m) / s)
    return np.vstack(arr).T

def load_artifacts(model_dir: Path):
    le = joblib.load(model_dir / "label_encoder.joblib")
    classes = json.load(open(model_dir / "classes.json"))
    model_type = open(model_dir / "model_type.txt").read().strip()

    scaler = None
    if (model_dir / "scaler.json").exists():
        scaler = json.load(open(model_dir / "scaler.json"))

    if model_type == "sklearn":
        pipe = joblib.load(model_dir / "best_model.joblib")
        return {"type": "sklearn", "pipe": pipe, "le": le, "classes": classes, "scaler": scaler}

    if not TORCH_OK:
        raise RuntimeError("Best model is DL but torch is not available.")
    meta = json.load(open(model_dir / "model_meta.json"))
    arch = meta.get("arch", "MLP"); in_dim = meta["in_dim"]; n_classes = meta["n_classes"]

    if arch == "MLP":
        model = MLPNet(in_dim, n_classes)
    elif arch == "LSTM":
        model = LSTMNet(in_dim, n_classes, hid=64)
    elif arch == "Transformer":
        model = TransformerTab(in_dim, n_classes, d_model=64, nhead=4, nlayers=2, dim_ff=128, dropout=0.1)
    else:
        raise RuntimeError(f"Unknown DL arch in model_meta.json: {arch}")

    state = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state); model.eval()
    return {"type": "dl", "model": model, "le": le, "classes": classes, "scaler": scaler, "arch": arch}

def predict_with_loaded(loaded, Xdf: pd.DataFrame):
    if loaded["type"] == "sklearn":
        pipe = loaded["pipe"]
        try:
            proba = pipe.predict_proba(Xdf)
        except Exception:
            proba = None
        pred_idx = pipe.predict(Xdf)
    else:
        if loaded["scaler"] is None:
            raise RuntimeError("DL model requires scaler.json")
        X_std = apply_standardizer(Xdf, loaded["scaler"])
        xt = torch.tensor(X_std, dtype=torch.float32)
        with torch.no_grad():
            logits = loaded["model"](xt)
            proba_t = torch.softmax(logits, dim=1)
            proba = proba_t.numpy()
        pred_idx = np.argmax(proba, axis=1)

    pred_lbl = loaded["le"].inverse_transform(pred_idx)
    return pred_lbl, proba

# -------------------- Language helpers --------------------
LANG_CHOICES = [
    "English",
    "Hindi (हिन्दी)",
    "Kannada (ಕನ್ನಡ)",
    "Telugu (తెలుగు)",
    "Tamil (தமிழ்)",
    "Malayalam (മലയാളം)",
]
LANG_TO_SCRIPT = {
    "English":   "Latin",
    "Hindi":     "Devanagari",
    "Kannada":   "Kannada",
    "Telugu":    "Telugu",
    "Tamil":     "Tamil",
    "Malayalam": "Malayalam",
}
SCRIPT_RANGES = {
    "Latin":      [(0x0041,0x007A)],
    "Devanagari": [(0x0900,0x097F)],
    "Kannada":    [(0x0C80,0x0CFF)],
    "Telugu":     [(0x0C00,0x0C7F)],
    "Tamil":      [(0x0B80,0x0BFF)],
    "Malayalam":  [(0x0D00,0x0D7F)],
}
TTS_LANG_CODE = {  # gTTS language codes
    "English":   "en",
    "Hindi":     "hi",
    "Kannada":   "kn",
    "Telugu":    "te",
    "Tamil":     "ta",
    "Malayalam": "ml",
}

def _lang_display_to_name(s: str) -> str:
    if s.startswith("Hindi"): return "Hindi"
    if s.startswith("Kannada"): return "Kannada"
    if s.startswith("Telugu"): return "Telugu"
    if s.startswith("Tamil"): return "Tamil"
    if s.startswith("Malayalam"): return "Malayalam"
    return "English"

def script_coverage(text: str, script: str) -> float:
    ranges = SCRIPT_RANGES.get(script, [])
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    def in_ranges(cp): return any(lo <= cp <= hi for lo,hi in ranges)
    count = sum(1 for ch in letters if in_ranges(ord(ch)))
    return count / max(1, len(letters))

# -------------------- Ollama calls --------------------
def ollama_chat(model: str, system: str, user: str, timeout=180) -> str:
    try:
        payload = {"model": model, "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user},
        ], "stream": False}
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
        if r.status_code == 200:
            return (r.json().get("message",{}) or {}).get("content","").strip()
        return f"(Ollama status {r.status_code}: {r.text[:200]})"
    except Exception as e:
        return f"(Ollama call failed: {e})"

def ask_ollama_recos(ollama_model: str, crop: str, row: Dict[str, Any]) -> str:
    system = ("You are an agronomy assistant. Reply in clean Markdown with section headers and short, safe bullets.")
    user = (
        f"Predicted crop: {crop}\n"
        f"Profile: N={row['Nitrogen']} P={row['Phosphorus']} K={row['Potassium']}, "
        f"pH={row['pH_Value']}, Temp={row['Temperature']}°C, "
        f"Humidity={row['Humidity']}%, Rainfall={row['Rainfall']} mm\n\n"
        "Write 5–7 precise recommendations using these sections:\n"
        "### NPK & Timing\n- ...\n"
        "### pH Management\n- ...\n"
        "### Irrigation\n- ...\n"
        "### Crop-Specific Practices\n- ...\n"
        "### Extra Tips\n- ...\n"
    )
    return ollama_chat(ollama_model, system, user)

def ollama_translate_markdown(ollama_model: str, text_en: str, target_language: str) -> str:
    system = ("You are a precise translator. Translate the user's Markdown content exactly into the requested language. "
              "Preserve headings, lists, numbers, units, and formatting. Do NOT include any explanation or English.")
    user = (f"Translate the following Markdown into {target_language}.\n\n"
            "Do not add or remove content. Only translate.\n\n"
            f"---\n{text_en}\n---")
    return ollama_chat(ollama_model, system, user)

def ensure_language_script(text: str, language: str) -> bool:
    if language == "English":
        return True
    script = LANG_TO_SCRIPT.get(language, "Latin")
    cov = script_coverage(text, script)
    return cov >= 0.6

# -------------------- TTS --------------------
def strip_markdown(md: str) -> str:
    """Minimal markdown strip for TTS clarity."""
    # remove code fences & inline backticks
    text = re.sub(r"```.*?```", "", md, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # remove markdown headings / bullets symbols only (keep content)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def tts_gtts(text: str, language_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (audio_path, error_msg)."""
    if not GTTS_OK:
        return None, "gTTS not installed. Run: pip install gTTS"
    lang_code = TTS_LANG_CODE.get(language_name, "en")
    try:
        clean = strip_markdown(text)
        # gTTS writes mp3; Gradio can play it directly
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts = gTTS(clean, lang=lang_code, slow=False)
        tts.save(tmp.name)
        return tmp.name, None
    except Exception as e:
        return None, f"TTS error: {e}"

# -------------------- Images --------------------
def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def build_image_index(images_dir: Path) -> Dict[str, Path]:
    idx = {}
    if not images_dir.exists():
        print(f"[images] Directory not found: {images_dir.resolve()}")
        return idx
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    files = [p for p in images_dir.rglob("*") if p.suffix in exts and p.is_file()]
    if not files:
        print(f"[images] No images found under: {images_dir.resolve()}")
    for p in files:
        key = _norm_key(p.stem)
        idx.setdefault(key, p)
    print(f"[images] Indexed {len(idx)} images from {images_dir.resolve()}")
    print(f"[images] Sample keys: {list(idx.keys())[:10]}")
    return idx

def find_crop_image(crop: str, index: Dict[str, Path]) -> Optional[Image.Image]:
    key = _norm_key(crop)
    p = index.get(key)
    if not p:
        print(f"[images] No image for '{crop}' (key={key})")
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as e:
        print(f"[images] Failed to open {p}: {e}")
        return None

def placeholder_image(crop: str) -> Image.Image:
    img = Image.new("RGB", (768, 480), color=(218, 237, 222))
    d = ImageDraw.Draw(img)
    d.text((24, 24), crop, fill=(25, 76, 51))
    return img

# -------------------- Sample row select (Gradio v4) --------------------
def on_sample_select(df: pd.DataFrame, evt: gr.SelectData):
    if evt is None or evt.index is None:
        return [gr.update()]*7
    ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        ridx = int(ridx)
        if ridx < 0 or ridx >= len(df):
            return [gr.update()]*7
        row = df.iloc[ridx]
        vals = row[["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"]].tolist()
        return vals
    except Exception:
        return [gr.update()]*7

# -------------------- UI --------------------
CUSTOM_CSS = """
:root { --bg:#E8F5E9; --card:#ffffff; --accent:#1B5E20; }
.gradio-container {background: var(--bg);}
h1.title { text-align:center; color:var(--accent); font-weight:800; margin-bottom:18px; }
.card { background:var(--card); padding:14px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
#reco-md * { font-size:16px !important; line-height:1.55 !important; }
.reco-wrap { max-height:800px; overflow:auto; padding-right:8px; }
.num-compact input { padding:8px 10px !important; }
.smallnote { color:#2d5937; font-size:12px; margin-top:6px; }
"""

def build_interface(model_dir: Path, images_dir: Path, ollama_model: str, port: int):
    loaded = load_artifacts(model_dir)
    img_index = build_image_index(images_dir)
    print(f"[images] Using images_dir = {images_dir.resolve()}")

    # Presets
    examples = [
        [90, 42, 43, 26, 85, 6.5, 220],
        [80, 40, 40, 26, 75, 6.5, 120],
        [50, 50, 50, 24, 60, 6.8, 80],
        [20, 30, 20, 22, 55, 6.2, 90],
        [30, 50, 120, 28, 70, 6.0, 150],
    ]
    example_df = pd.DataFrame(examples, columns=["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH","Rainfall"])

    def predict_fn(N, P, K, T, H, pH, R, ask_llm, language_display, speak):
        # features
        row = {"Nitrogen": float(N),"Phosphorus": float(P),"Potassium": float(K),
               "Temperature": float(T),"Humidity": float(H),"pH_Value": float(pH),"Rainfall": float(R)}
        dfX = pd.DataFrame([row], columns=NUMERIC_FEATURES)

        # model prediction
        pred_lbl, proba = predict_with_loaded(loaded, dfX)
        crop = pred_lbl[0]

        # top-2 probabilities
        if proba is None:
            top_df = pd.DataFrame([{"Rank":"Top-1","Class":crop,"Prob":1.0},{"Rank":"Top-2","Class":"","Prob":""}],
                                  columns=["Rank","Class","Prob"])
        else:
            classes = np.array(loaded["classes"])
            order = np.argsort(-proba[0])[:2]
            rows = [{"Rank":f"Top-{i+1}","Class":classes[idx],"Prob":round(float(proba[0,idx]),4)} for i,idx in enumerate(order)]
            while len(rows)<2: rows.append({"Rank":f"Top-{len(rows)+1}","Class":"","Prob":""})
            top_df = pd.DataFrame(rows, columns=["Rank","Class","Prob"])

        # image
        img = find_crop_image(crop, img_index) or placeholder_image(crop)

        # LLM recos (two-pass for language reliability)
        language = _lang_display_to_name(language_display)
        if not ask_llm:
            reco_text = "(Recommendations disabled)"
        else:
            base_en = ask_ollama_recos(ollama_model, crop, row)  # English first
            if language == "English":
                reco_text = base_en
            else:
                translated = ollama_translate_markdown(ollama_model, base_en, language)
                if not ensure_language_script(translated, language):
                    # Retry once with stronger hint
                    translated = ollama_translate_markdown(
                        ollama_model,
                        "IMPORTANT: Do not include any English. Translate strictly.\n\n" + base_en,
                        language
                    )
                reco_text = translated

        # TTS (optional)
        audio_path = None
        tts_note = ""
        if speak:
            audio_path, err = tts_gtts(reco_text, language)
            if err:
                tts_note = f"⚠️ {err}"

        # For gr.Audio you can return a filepath; Gradio will load the file.
        return crop, top_df, img, reco_text, (audio_path if audio_path else None), tts_note

    # ----------- Layout -----------
    with gr.Blocks(css=CUSTOM_CSS, title="Crop Prediction & Recommendations") as demo:
        gr.HTML("<h1 class='title'>🌾 AI-Powered Multilingual Crop Prediction with Voice Recommendations</h1>")

        with gr.Row():
            # Left
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.Markdown("**Input Soil & Climate Parameters**")

                gr.Markdown("**Soil Inputs**")
                with gr.Row(equal_height=True):
                    N   = gr.Number(label="Nitrogen (N)",  value=80, elem_classes=["num-compact"])
                    P   = gr.Number(label="Phosphorus (P)",value=40, elem_classes=["num-compact"])
                    K   = gr.Number(label="Potassium (K)", value=40, elem_classes=["num-compact"])
                    pHv = gr.Number(label="pH",            value=6.5,elem_classes=["num-compact"])

                gr.Markdown("**Climate Inputs**")
                with gr.Row(equal_height=True):
                    T = gr.Number(label="Temperature (°C)", value=26, elem_classes=["num-compact"])
                    H = gr.Number(label="Humidity (%)",     value=75, elem_classes=["num-compact"])
                    R = gr.Number(label="Rainfall (mm)",    value=120,elem_classes=["num-compact"])

                with gr.Row():
                    ask   = gr.Checkbox(label="Generate recommendations", value=True)
                    lang  = gr.Dropdown(choices=LANG_CHOICES, value="English", label="Recommendation Language")
                with gr.Row():
                    speak = gr.Checkbox(label="Speak recommendations (Text-to-Speech)", value=False)

                btn = gr.Button("Predict", variant="primary")

                gr.Markdown("**Sample inputs (click a row):**")
                ex_table = gr.Dataframe(value=example_df,
                                        headers=list(example_df.columns),
                                        datatype=["number"]*7,
                                        interactive=True)
                ex_table.select(fn=on_sample_select, inputs=[ex_table], outputs=[N,P,K,T,H,pHv,R])

            # Middle
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.Markdown("**Prediction & Confidence**")
                pred_crop  = gr.Textbox(label="Predicted Crop", interactive=False)
                conf_table = gr.Dataframe(label="Top Probabilities (Top-1 & Top-2)", interactive=False)
                img_out    = gr.Image(label="Crop Image", type="pil")

            # Right
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.Markdown("**Yield Recommendations**")
                reco = gr.Markdown("Toggle ‘Generate recommendations’ and choose a language.", elem_id="reco-md", elem_classes=["reco-wrap"])
                audio = gr.Audio(label="Voice (TTS)", interactive=False, autoplay=False)
                tts_note = gr.Markdown("", elem_classes=["smallnote"])

        btn.click(
            predict_fn,
            inputs=[N,P,K,T,H,pHv,R,ask,lang,speak],
            outputs=[pred_crop,conf_table,img_out,reco,audio,tts_note]
        )

    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ollama_model", default="llama3")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_interface(Path(args.model_dir), Path(args.images_dir), args.ollama_model, args.port)

if __name__ == "__main__":
    main()


#gTTS needs internet access. If you want a fully offline option later,  wireing in Piper TTS or Coqui TTS models for Hindi/Tamil/etc. (heavier download, but works offline).