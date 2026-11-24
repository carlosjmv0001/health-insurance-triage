#!/usr/bin/env python3
"""Download official raw datasets for the underwriting triage study.

Sources:
1. Spanish Health Insurance Portfolio (SHIP; Lledo, Espinosa & Perez, Scientific Data, 2026;
   Mendeley Data DOI: 10.17632/386vmj2tbk.4, CC BY 4.0)
2. MEPS-HC Longitudinal Panels (AHRQ, public use files: Panels 22, 23, 25, 26, 27, 28)
   - h210: Panel 22 (2017-2018)
   - h217: Panel 23 (2018-2019)
   - h236: Panel 25 (2020-2021)
   - h244: Panel 26 (2021-2022)
   - h252: Panel 27 (2022-2023)
   - h257: Panel 28 (2023-2024)
3. Medical Insurance Cost Prediction (MICP; synthetic audit benchmark hosted on Kaggle:
   mohankrishnathalla/medical-insurance-cost-prediction, CC0)
"""
from pathlib import Path
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MEPS_PANELS = ["h210", "h217", "h236", "h244", "h252", "h257"]


def download_spanish_portfolio():
    out = DATA / "spanish_portfolio"
    out.mkdir(parents=True, exist_ok=True)
    main_xlsx = out / "Dataset of health insurance portfolio.xlsx"
    if main_xlsx.exists() and main_xlsx.stat().st_size > 10_000_000:
        print("[Spanish Portfolio] Already present, skipping download.")
        return

    meta_path = out / "_mendeley_meta.json"
    subprocess.check_call(
        ["curl", "-sL", "https://data.mendeley.com/public-api/datasets/386vmj2tbk", "-o", str(meta_path)]
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for f in meta["files"]:
        dest = out / f["filename"]
        if not dest.exists():
            print(f"[Spanish Portfolio] Downloading {f['filename']} ({f['size']/1e6:.2f} MB)...")
            subprocess.check_call(["curl", "-sL", f["content_details"]["download_url"], "-o", str(dest)])
    meta_path.unlink(missing_ok=True)
    print("[Spanish Portfolio] Done.")


def download_meps_panels():
    out = DATA / "meps"
    out.mkdir(parents=True, exist_ok=True)
    for code in MEPS_PANELS:
        dta = out / f"{code}.dta"
        if dta.exists() and dta.stat().st_size > 1_000_000:
            print(f"[MEPS {code}] Already present ({dta.stat().st_size/1e6:.1f} MB), skipping.")
            continue
        zip_path = out / f"{code}dta.zip"
        url = f"https://meps.ahrq.gov/mepsweb/data_files/pufs/{code}/{code}dta.zip"
        print(f"[MEPS {code}] Downloading from {url}...")
        subprocess.check_call(["curl", "-fsSL", url, "-o", str(zip_path)])
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out)
        print(f"[MEPS {code}] Extracted -> {dta.name} ({dta.stat().st_size/1e6:.1f} MB)")


def download_kaggle_synthetic():
    out = DATA / "kaggle"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "medical_insurance.csv"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"[Kaggle] Already present ({target.stat().st_size/1e6:.1f} MB), skipping.")
        return
    try:
        import kagglehub

        print("[Kaggle] Downloading mohankrishnathalla/medical-insurance-cost-prediction via kagglehub...")
        path = Path(kagglehub.dataset_download("mohankrishnathalla/medical-insurance-cost-prediction"))
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {path}")
        shutil.copy2(csvs[0], target)
        print(f"[Kaggle] Saved -> {target} ({target.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"[Kaggle] Automatic download failed: {e}")
        print(f"         Place medical_insurance.csv manually in {target} if needed.")


if __name__ == "__main__":
    download_spanish_portfolio()
    download_meps_panels()
    download_kaggle_synthetic()
