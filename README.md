# Margin Definition Over Model Complexity in Health Insurance Triage

Official reproducibility repository for the manuscript *"Margin Definition Over Model Complexity in Health Insurance Triage"*.

This repository contains the complete data-retrieval, modeling, sensitivity-audit, fairness-audit, and figure-generation pipeline across three complementary individual-level health insurance datasets totaling over 350,000 records:

| Study | Dataset | Scope / Size | Official Source & License |
| :--- | :--- | :--- | :--- |
| **Study 1** | **MICP** (*Medical Insurance Cost Prediction*) | Synthetic benchmark; $N = 100{,}000$ | [Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction) (CC0: Public Domain) |
| **Study 2** | **SHIP** (*Spanish Health Insurance Portfolio*) | Real Spanish private health insurer; $228{,}711$ policy-years (2017–2019; $100{,}453$ unique insureds) | [Mendeley Data](https://doi.org/10.17632/386vmj2tbk.4) (CC BY 4.0; Lledó, Espinosa & Pérez, *Scientific Data*, 2026) |
| **Study 3** | **MEPS-HC** (*Medical Expenditure Panel Survey—Household Component*) | Six pooled two-year U.S. longitudinal panels (Panels 22, 23, 25, 26, 27, 28; 2017–2024; $N = 21{,}490$ continuously privately insured adults) | [AHRQ MEPS](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp) (U.S. Public Domain) |

Raw data files are **not** redistributed in this repository; they are downloaded directly from their official providers via `scripts/download_data.py`. Pre-computed experimental outputs (`results/`) and 300-dpi manuscript figures (`figures/`) are included for immediate inspection and verification.

---

## Repository Structure

```text
health-insurance-triage/
├── scripts/
│   ├── download_data.py    # Downloads SHIP, MEPS-HC panels 22-28, and MICP into data/
│   ├── common.py           # Unified evaluation engine (11 learners, validation tuning,
│   │                       # paired bootstrap, calibration, manual review, fairness)
│   ├── exp1_kaggle.py      # Study 1 (MICP): D0/D2/D1 margins, 20 splits, F1/F2/F3 ablation
│   ├── exp2_spanish.py     # Study 2 (SHIP): New Business (2A) & Renewal (2B), lambda 0-35%, 20 ID-disjoint splits
│   ├── exp3_meps.py        # Study 3 (MEPS-HC): Clinical (3A) & Renewal (3B), CPI-U 2024 USD, 20 splits
│   └── make_figures.py     # Generates all 5 publication-grade 300-dpi figures from results/
├── results/
│   ├── exp1/               # Study 1 CSVs, rule texts, and execution log
│   ├── exp2/               # Study 2 CSVs, rule texts, and execution log
│   └── exp3/               # Study 3 CSVs, rule texts, and execution log
├── figures/
│   ├── fig1_margin_sensitivity.png
│   ├── fig2_model_comparison_20splits.png
│   ├── fig3_calibration_and_operating_points.png
│   ├── fig4_manual_review_sensitivity.png
│   └── fig5_fairness_and_ablation.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup & Reproduction

### 1. Environment
Tested with Python 3.11+. Install dependencies in a clean virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: `scripts/download_data.py` uses `curl` to fetch SHIP from Mendeley Data and MEPS-HC `.dta` archives from AHRQ, and `kagglehub` for MICP. If `kagglehub` is not authenticated, download `medical_insurance.csv` manually from Kaggle and place it in `data/kaggle/medical_insurance.csv`.)*

### 2. Full End-to-End Execution

```bash
# 1. Download official raw datasets into data/
python3 scripts/download_data.py

# 2. Run all three empirical studies (writes CSVs and logs into results/)
python3 scripts/exp1_kaggle.py
python3 scripts/exp2_spanish.py
python3 scripts/exp3_meps.py

# 3. Regenerate all five manuscript figures in figures/
python3 scripts/make_figures.py
```

To regenerate the figures directly from the included CSV results without re-running model training:

```bash
python3 scripts/make_figures.py
```

---

## Mapping to Manuscript Tables and Figures

| Manuscript Item | Generating Script | Primary Output File(s) |
| :--- | :--- | :--- |
| **Table 2** (MICP accounting by deductible) | `exp1_kaggle.py` | `results/exp1/exp1_dataset_structure.csv` |
| **Table 3** (Margin sensitivity across all 3 datasets) | `exp1`, `exp2`, `exp3` | `*_protocol.csv` |
| **Table 4** (20-split cross-partition stability) | `exp1`, `exp2`, `exp3` | `*_20splits.csv` |
| **Table 5** (MICP six-rule financial decomposition) | `exp1_kaggle.py` | `results/exp1/exp1_rule_breakdown.csv` |
| **Table 6** (Manual review break-even audit costs) | `exp1`, `exp2`, `exp3` | `*_manual_review.csv` |
| **Figures 1–5** | `make_figures.py` | `figures/fig1_*.png` – `figures/fig5_*.png` |

---

## License

Code in this repository is released under the [MIT License](LICENSE). Original datasets remain subject to their respective providers' licenses (CC0 for MICP, CC BY 4.0 for SHIP, and U.S. Public Domain for MEPS-HC).
