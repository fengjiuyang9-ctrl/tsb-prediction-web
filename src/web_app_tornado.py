import argparse
import base64
import html
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import tornado.ioloop
import tornado.web
from PIL import Image

from dataset import MetaProcessor, build_transforms
from embedded_examples import EMBEDDED_EXAMPLES
from model import E2Model
from utils import StandardScaler1D


DEFAULT_RUN_DIR = Path(
    "experiments_实验输出/e2_final_F3_reg_lr_tune_5fold10/raw_5fold10_F3_reg_lr_tune/run_20260417_223410"
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


@dataclass
class FoldPredictor:
    fold_name: str
    model: E2Model
    transform: object
    meta_processor: MetaProcessor
    target_scaler: StandardScaler1D
    target_standardize: bool
    device: torch.device
    cfg: Dict

    @torch.no_grad()
    def predict_umol(self, image: Image.Image, age_hours: float, sex: str, race: str) -> float:
        x = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
        row = pd.Series(
            {
                self.cfg["age_col"]: float(age_hours),
                self.cfg["sex_col"]: str(sex),
                self.cfg["race_col"]: str(race),
            }
        )
        meta_np = self.meta_processor.transform_row(row)
        m = torch.tensor(meta_np, dtype=torch.float32).unsqueeze(0).to(self.device, non_blocking=True)
        pred_model = float(self.model(x, m).detach().cpu().numpy()[0])
        if self.target_standardize:
            return float(self.target_scaler.inverse_transform(np.array([pred_model], dtype=np.float64))[0])
        return pred_model


def build_fold_predictor(fold_dir: Path, cfg: Dict, device: torch.device) -> FoldPredictor:
    train_csv = fold_dir / "split_train.csv"
    ckpt = fold_dir / "best_model.pt"
    if not train_csv.exists():
        raise FileNotFoundError(f"missing split_train.csv: {train_csv}")
    if not ckpt.exists():
        raise FileNotFoundError(f"missing best_model.pt: {ckpt}")

    train_df = pd.read_csv(train_csv, encoding="utf-8-sig")
    meta_processor = MetaProcessor(
        age_col=cfg["age_col"],
        sex_col=cfg["sex_col"],
        race_col=cfg["race_col"],
        weight_col=cfg["weight_col"],
        use_weight=bool(cfg["use_weight"]),
        sex_categories=list(cfg["sex_categories"]),
        race_categories=list(cfg["race_categories"]),
        extra_numeric_cols=list(cfg.get("extra_numeric_meta_cols", [])),
    )
    meta_processor.fit(train_df)

    target_scaler = StandardScaler1D()
    if bool(cfg["target_standardize"]):
        vals = pd.to_numeric(train_df[cfg["target_col"]], errors="coerce").fillna(0.0).values
        target_scaler.fit(vals)

    model = E2Model(
        meta_input_dim=meta_processor.output_dim,
        meta_hidden=int(cfg["meta_hidden"]),
        meta_out=int(cfg["meta_out"]),
        head_hidden=int(cfg["head_hidden"]),
        dropout=float(cfg["dropout"]),
        use_imagenet_pretrained=False,
    )
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return FoldPredictor(
        fold_name=fold_dir.name,
        model=model,
        transform=build_transforms(int(cfg["image_size"]), is_train=False),
        meta_processor=meta_processor,
        target_scaler=target_scaler,
        target_standardize=bool(cfg["target_standardize"]),
        device=device,
        cfg=cfg,
    )


def load_bundle(run_dir: Path) -> Dict:
    cfg = load_json(run_dir / "config_copy.json")
    cv = load_json(run_dir / "cv_summary.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dirs = sorted([p for p in run_dir.glob("fold_*") if p.is_dir()])
    max_folds_env = os.environ.get("MAX_FOLDS", "").strip()
    if max_folds_env:
        try:
            max_folds = int(max_folds_env)
            if max_folds > 0:
                fold_dirs = fold_dirs[:max_folds]
        except ValueError:
            pass
    predictors = [build_fold_predictor(fd, cfg, device) for fd in fold_dirs]
    if not predictors:
        raise RuntimeError(f"No predictors found in {run_dir}")
    return {"cfg": cfg, "cv": cv, "device": device, "predictors": predictors, "run_dir": run_dir}


def get_risk_thresholds(age_hours: float) -> Dict[str, float | str]:
    # Strict interval boundaries from your risk table:
    # 0<=age<24, 24<=age<48, 48<=age<=72, age>72
    # High-risk thresholds: 224/265/293/310; low-risk threshold fixed at 170
    if age_hours < 0:
        raise ValueError("Age (hours) cannot be negative.")
    if age_hours < 24:
        photo, high, band = 275.0, 224.0, "0-24"
    elif age_hours < 48:
        photo, high, band = 316.0, 265.0, "24-48"
    elif age_hours <= 72:
        photo, high, band = 344.0, 293.0, "48-72"
    else:  # age_hours > 72
        photo, high, band = 361.0, 310.0, ">72"
    return {"band": band, "photo": photo, "high": high, "low": 170.0}


def classify_risk(tsb_umol: float, age_hours: float) -> Dict[str, float | str]:
    th = get_risk_thresholds(age_hours)
    low = float(th["low"])
    high = float(th["high"])
    if tsb_umol >= high:
        return {
            "level": "High Risk",
            "css": "badge-high",
            "advice": "Near or within high-risk zone. Prompt clinical evaluation is recommended.",
            "high": high,
            "low": low,
            "band": str(th["band"]),
        }
    if tsb_umol < low:
        return {
            "level": "Low Risk",
            "css": "badge-low",
            "advice": "Currently in low-risk zone. Routine follow-up per clinical guidance is suggested.",
            "high": high,
            "low": low,
            "band": str(th["band"]),
        }
    return {
        "level": "Intermediate Risk",
        "css": "badge-mid",
        "advice": "In the intermediate range. Close follow-up with clinical context is recommended.",
        "high": high,
        "low": low,
        "band": str(th["band"]),
    }


def aggregate_multi_image(values: List[float], method: str = "median") -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("No valid image predictions found.")
    if method == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def _collect_images_from_zip(zip_bytes: bytes, max_images: int = 200) -> List[Dict[str, bytes | str]]:
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    out: List[Dict[str, bytes | str]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            suffix = Path(name).suffix.lower()
            if suffix not in allowed:
                continue
            data = zf.read(info)
            if not data:
                continue
            out.append({"filename": Path(name).name, "body": data})
            if len(out) >= max_images:
                break
    return out


def render_page(bundle: Dict, result: Dict | None = None, error: str | None = None) -> str:
    examples = EMBEDDED_EXAMPLES
    gallery_html = "".join(
        (
            '<figure class="eg-card">'
            f'<div class="eg-illus"><img src="{e["data_uri"]}" alt="Jaundice visual reference" loading="lazy"/></div>'
            f'<figcaption><strong>{html.escape(e["title"])}</strong><br/>{html.escape(e["desc"])}</figcaption>'
            "</figure>"
        )
        for e in examples
    )
    examples_block = f"""
    <section class="card">
      <h2 class="sec-title">Visual Examples</h2>
      <p class="muted">Built-in reference illustrations for recommended check areas and pressing technique.</p>
      <div class="eg-grid">{gallery_html}</div>
    </section>
    """

    result_html = ""
    if error:
        result_html = f'<div class="card error">{html.escape(error)}</div>'
    elif result is not None:
        rows = "\n".join(
            (
                "<tr>"
                f"<td>{html.escape(str(r['name']))}</td>"
                f"<td>{float(r['pred']):.2f}</td>"
                f"<td>{float(r['std']):.2f}</td>"
                "</tr>"
            )
            for r in result["image_rows"]
        )
        img_html = ""
        if result.get("image_b64"):
            img_html = f'<img class="preview" src="data:image/jpeg;base64,{result["image_b64"]}" alt="uploaded" />'
        result_html = f"""
        <div class="card result">
          <h2>Result</h2>
          <p class="big">{result["final_pred"]:.2f} μmol/L</p>
          <p class="summary-line"><strong>Predicted TSB:</strong> {result["final_pred"]:.2f} μmol/L</p>
          <p class="summary-line"><strong>Risk Level:</strong> {result["risk_level"]}</p>
          <p class="summary-line"><strong>Recommendation:</strong> {html.escape(result["risk_advice"])}</p>
          <div class="risk-line">
            <span class="badge {result["risk_css"]}">{result["risk_level"]}</span>
            <span class="muted">Age band: {result["age_band"]} hours. Thresholds: Low Risk &lt; {result["low_thr"]:.0f}; High Risk ≥ {result["high_thr"]:.0f} μmol/L</span>
          </div>
          <p class="muted">Multi-image aggregation: {result["agg_method"]}; images: {result["n_images"]}; inter-image spread: {result["img_std"]:.2f}</p>
          {img_html}
          <table>
            <thead><tr><th>Image</th><th>Prediction (μmol/L)</th><th>Fold Std</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
          <div class="result-actions">
            <a class="link-btn" href="/">Try Another Image</a>
          </div>
        </div>
        <div class="card">
          <h3>Risk Threshold Reference</h3>
          <table>
            <thead>
              <tr><th>Age (h)</th><th>Phototherapy Threshold P(age)</th><th>High-Risk Threshold</th><th>Low-Risk Threshold</th><th>Intermediate Range</th></tr>
            </thead>
            <tbody>
              <tr><td>0-24</td><td>275</td><td>≥224</td><td>&lt;170</td><td>170 ~ 224</td></tr>
              <tr><td>24-48</td><td>316</td><td>≥265</td><td>&lt;170</td><td>170 ~ 265</td></tr>
              <tr><td>48-72</td><td>344</td><td>≥293</td><td>&lt;170</td><td>170 ~ 293</td></tr>
              <tr><td>&gt;72</td><td>361</td><td>≥310</td><td>&lt;170</td><td>170 ~ 310</td></tr>
            </tbody>
          </table>
          <p class="muted">This result is for reference only and does not replace clinical diagnosis.</p>
        </div>
        """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>TSB Prediction System</title>
  <style>
    :root {{
      --bg1: #f7f5ef;
      --bg2: #e8f0ed;
      --bg3: #f0f4f8;
      --card: #ffffff;
      --ink: #24323d;
      --muted: #5e6a74;
      --danger: #b91c1c;
      --line: #d6dde3;
      --soft: #eef3f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "IBM Plex Sans", "Noto Sans SC", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 620px at 80% -10%, #dcefe7 0%, transparent 58%),
                  radial-gradient(1100px 500px at -15% 5%, #f4eee3 0%, transparent 52%),
                  linear-gradient(140deg, var(--bg1), var(--bg2) 60%, var(--bg3));
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 920px;
      margin: 0 auto;
      padding: 18px 14px 36px;
    }}
    .hero {{ margin-bottom: 12px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(24px, 5vw, 38px); line-height: 1.1; letter-spacing: .2px; }}
    .sec-title {{ margin: 0 0 8px; font-size: clamp(18px, 3.5vw, 24px); line-height: 1.2; }}
    .sub {{ font-size: 14px; color: var(--muted); margin-top: 4px; }}
    .card {{
      background: linear-gradient(180deg, #ffffff, #fcfdff);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      margin-top: 12px;
      box-shadow: 0 10px 28px rgba(36, 50, 61, 0.08);
      animation: fadeIn .45s ease;
    }}
    .error {{ border-color: #fecaca; color: var(--danger); }}
    form {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .full {{ grid-column: 1 / -1; }}
    label {{ font-size: 13px; color: var(--muted); display: block; margin-bottom: 4px; }}
    input, select {{
      width: 100%;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      font-size: 15px;
      background: #fff;
      transition: border-color .2s ease, box-shadow .2s ease;
    }}
    input:focus, select:focus {{
      outline: none;
      border-color: #8ab4a4;
      box-shadow: 0 0 0 3px rgba(138, 180, 164, .18);
    }}
    button {{
      width: 100%;
      border: 0;
      border-radius: 12px;
      background: linear-gradient(135deg, #2d7f74, #3a8a7e);
      color: #fff;
      font-size: 16px;
      font-weight: 700;
      padding: 11px 12px;
      cursor: pointer;
      transition: transform .12s ease, box-shadow .2s ease, filter .2s ease;
    }}
    button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 8px 18px rgba(45, 127, 116, .28);
      filter: brightness(1.02);
    }}
    .file-native {{
      position: absolute;
      left: -9999px;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }}
    .file-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      min-height: 44px;
    }}
    .file-btn {{
      border: 1px solid #8aa9b7;
      border-radius: 8px;
      background: linear-gradient(180deg, #f8fbff, #edf4fa);
      color: #27465a;
      padding: 7px 12px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
    }}
    .file-label {{
      font-size: 14px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }}
    .result h2 {{ margin: 0 0 8px; }}
    .result h3 {{ margin: 0 0 8px; }}
    .big {{ font-size: 30px; font-weight: 800; margin: 0; color: #0f5132; }}
    .summary-line {{ margin: 4px 0; font-size: 15px; color: var(--ink); }}
    .muted {{ color: var(--muted); margin: 6px 0 10px; font-size: 13px; }}
    .risk-line {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 8px 0 6px; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 6px 12px; color: #fff; font-weight: 700; font-size: 14px; }}
    .badge-high {{ background: #b91c1c; }}
    .badge-mid {{ background: #c2410c; }}
    .badge-low {{ background: #15803d; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }}
    .preview {{
      width: 100%;
      max-height: 280px;
      object-fit: contain;
      border-radius: 10px;
      border: 1px solid var(--line);
      margin: 8px 0 10px;
      background: #f8fafc;
    }}
    .eg-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }}
    .eg-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: var(--soft);
    }}
    .eg-illus {{
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #edf2f6;
    }}
    .eg-illus img {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
    }}
    .eg-card figcaption {{
      padding: 10px 12px 12px;
      font-size: 13px;
      line-height: 1.4;
      color: var(--muted);
    }}
    .optional-label {{
      font-size: 13px;
      color: var(--muted);
      font-weight: 500;
    }}
    .result-actions {{
      margin-top: 10px;
      display: flex;
      justify-content: flex-end;
    }}
    .link-btn {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 14px;
      text-decoration: none;
      color: #2d4c60;
      background: #f7fbff;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 720px) {{
      form {{ grid-template-columns: 1fr; }}
      .eg-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>TSB Prediction System</h1>
      <div class="sub">User mode: each image is predicted by {len(bundle["predictors"])}-fold ensemble, then multiple images are aggregated by median.</div>
    </section>
    {examples_block}

    <section class="card">
      <form action="/predict" method="post" enctype="multipart/form-data">
        <div>
          <label>Age (hours)</label>
          <input name="age_hours" type="number" min="0" max="2000" step="1" value="72" required />
        </div>
        <div>
          <label>Sex</label>
          <select name="sex" required>
            <option value="M">Male (M)</option>
            <option value="F">Female (F)</option>
            <option value="UNK">Unknown (UNK)</option>
          </select>
        </div>
        <div>
          <label>Race</label>
          <select name="race" required>
            <option value="Chinese">Chinese</option>
            <option value="Malay">Malay</option>
            <option value="Indian">Indian</option>
            <option value="Other">Other</option>
            <option value="UNK">UNK</option>
          </select>
        </div>
        <div class="full">
          <label>Upload Images</label>
          <div class="muted">Upload one or more neonatal skin images for prediction.</div>
          <div class="muted">Supported regions: forehead, chest, and foot.</div>
          <input id="images" class="file-native" name="images" type="file" accept=".jpg,.jpeg,.png,image/*" multiple />
          <div class="file-row">
            <button type="button" id="imagesBtn" class="file-btn">Choose Images</button>
            <span id="pickInfo" class="file-label">No files selected</span>
          </div>
          <div class="muted">Tip: Hold Ctrl or Shift to select multiple files on desktop.</div>
        </div>
        <div class="full">
          <label class="optional-label">Optional: Upload ZIP Batch</label>
          <div class="muted">Upload one ZIP file for batch prediction.</div>
          <input id="imagesZip" class="file-native" name="images_zip" type="file" accept=".zip,application/zip,application/x-zip-compressed" />
          <div class="file-row">
            <button type="button" id="zipBtn" class="file-btn">Choose ZIP File</button>
            <span id="zipInfo" class="file-label">No ZIP selected</span>
          </div>
        </div>
        <div class="full">
          <button type="submit">Run Prediction</button>
        </div>
        <div class="full">
          <a class="link-btn" href="/">Clear</a>
        </div>
      </form>
    </section>
    {result_html}
  </main>
  <script>
    (function () {{
      const input = document.getElementById('images');
      const info = document.getElementById('pickInfo');
      const imagesBtn = document.getElementById('imagesBtn');
      const zipInput = document.getElementById('imagesZip');
      const zipInfo = document.getElementById('zipInfo');
      const zipBtn = document.getElementById('zipBtn');
      if (imagesBtn && input) imagesBtn.addEventListener('click', function () {{ input.click(); }});
      if (zipBtn && zipInput) zipBtn.addEventListener('click', function () {{ zipInput.click(); }});
      if (input && info) input.addEventListener('change', function () {{
        const files = Array.from(input.files || []);
        if (!files.length) {{
          info.textContent = 'No files selected';
          return;
        }}
        const names = files.slice(0, 5).map(f => f.name).join('，');
        const more = files.length > 5 ? ` ... total ${{files.length}} files` : ` (total ${{files.length}})`;
        info.textContent = `Selected: ${{names}}${{more}}`;
      }});
      if (zipInput && zipInfo) zipInput.addEventListener('change', function () {{
        const f = (zipInput.files || [])[0];
        zipInfo.textContent = f ? `Selected ZIP: ${{f.name}}` : 'No ZIP selected';
      }});
    }})();
  </script>
</body>
</html>
"""


class BaseHandler(tornado.web.RequestHandler):
    @property
    def bundle(self) -> Dict:
        return self.application.settings["bundle"]


class HomeHandler(BaseHandler):
    def get(self):
        self.write(render_page(self.bundle))


class PredictHandler(BaseHandler):
    def post(self):
        try:
            age_hours = float(self.get_body_argument("age_hours"))
            sex = self.get_body_argument("sex").strip()
            race = self.get_body_argument("race").strip()
            if sex not in {"M", "F", "UNK"}:
                raise ValueError("Invalid sex value.")
            if race not in {"Chinese", "Malay", "Indian", "Other", "UNK"}:
                raise ValueError("Invalid race value.")

            files = self.request.files.get("images", [])
            if not files:
                files = self.request.files.get("images[]", [])
            if not files:
                files = self.request.files.get("image", [])
            zip_files = self.request.files.get("images_zip", [])
            if not zip_files:
                zip_files = self.request.files.get("imagesZip", [])

            extra_from_zip: List[Dict[str, bytes | str]] = []
            for z in zip_files:
                extra_from_zip.extend(_collect_images_from_zip(z["body"], max_images=200))

            if extra_from_zip:
                files = list(files) + extra_from_zip

            if not files:
                raise ValueError("Please upload at least one image.")
            if len(files) > 200:
                raise ValueError(f"At most 200 images per submission. Current: {len(files)}.")

            image_rows = []
            subject_preds = []
            first_img_b64 = ""

            for idx, f in enumerate(files, start=1):
                image = Image.open(io.BytesIO(f["body"])).convert("RGB")
                fold_preds = []
                for p in self.bundle["predictors"]:
                    v = p.predict_umol(image=image, age_hours=age_hours, sex=sex, race=race)
                    fold_preds.append(float(v))

                arr = np.asarray(fold_preds, dtype=np.float64)
                pred = float(np.mean(arr))
                std = float(np.std(arr))
                subject_preds.append(pred)
                name = f.get("filename") or f"image_{idx}"
                image_rows.append({"name": str(name), "pred": pred, "std": std})

                if idx == 1:
                    bio = io.BytesIO()
                    image.copy().thumbnail((900, 900))
                    image.save(bio, format="JPEG", quality=90)
                    first_img_b64 = base64.b64encode(bio.getvalue()).decode("ascii")

            final_pred = aggregate_multi_image(subject_preds, method="median")
            img_std = float(np.std(np.asarray(subject_preds, dtype=np.float64))) if len(subject_preds) > 1 else 0.0
            risk = classify_risk(final_pred, age_hours)

            self.write(
                render_page(
                    self.bundle,
                    result={
                        "final_pred": final_pred,
                        "risk_level": str(risk["level"]),
                        "risk_css": str(risk["css"]),
                        "risk_advice": str(risk["advice"]),
                        "high_thr": float(risk["high"]),
                        "low_thr": float(risk["low"]),
                        "age_band": str(risk["band"]),
                        "image_rows": image_rows,
                        "image_b64": first_img_b64,
                        "agg_method": "median",
                        "n_images": len(image_rows),
                        "img_std": img_std,
                    },
                )
            )
        except Exception as e:
            self.write(render_page(self.bundle, error=f"Prediction failed: {e}"))


def make_app(bundle: Dict) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", HomeHandler),
            (r"/predict", PredictHandler),
        ],
        bundle=bundle,
        debug=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TSB web app (tornado)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8501")))
    p.add_argument("--run_dir", type=str, default=os.environ.get("RUN_DIR", str(DEFAULT_RUN_DIR)))
    p.add_argument("--check_only", action="store_true", help="Load models and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    bundle = load_bundle(run_dir)
    if args.check_only:
        print(f"check_ok run_dir={run_dir} device={bundle['device']} folds={len(bundle['predictors'])}")
        return
    app = make_app(bundle)
    app.listen(args.port, address=args.host)
    print(f"TSB web app ready at http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"For mobile, open: http://<your-lan-ip>:{args.port}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()

