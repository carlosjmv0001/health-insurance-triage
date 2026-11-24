#!/usr/bin/env python3
"""Study 3: MEPS-HC Longitudinal Panels (AHRQ Panels 22, 23, 25, 26, 27, 28; 2017-2024).

Real-world individual-level US health insurance & expenditure cohort (N ~ 21,490 private adults aged 18-64
continuously covered by private insurance across both Year 1 and Year 2).

Strict temporal separation:
  - Features strictly from Year 1 (12 diagnosed chronic conditions + chronic_count, smoking, self-rated
    physical/mental health, demographics, socioeconomic status).
  - Realized financial outcome from Year 2: private-insurer paid claims (TOTPRVY2) and out-of-pocket (TOTSLFY2),
    inflation-adjusted to constant 2024 medical USD using official BLS CPI-U Medical Care annual averages.
  - Temporal holdout split: Panels 22-26 (2017-2022) = Train/Val (75/25), Panels 27-28 (2022-2024) = Test;
    plus 20 random 60/20/20 partitions for cross-split stability.

Two underwriting settings:
  3A. Clinical Underwriting (ex-ante Year 1 clinical + demographic features only; no prior expenditure).
  3B. Renewal Underwriting (adding Year 1 private claims TOTPRVY1 and utilization).

Four margin definitions (ex-ante benchmark premiums calibrated strictly on training data):
  - P_comm_80:     Uniform community-rated premium (80% target MLR) - TOTPRVY2
  - P_age_80:      Age-band rated premium (80% target MLR) - TOTPRVY2
  - P_age_95:      Age-band rated premium (95% target MLR, tight margin) - TOTPRVY2
  - P_age_80_D0err: Age-band rated premium (80% MLR) + TOTSLFY2 - TOTPRVY2
                    (mirrors the MICP D0 accounting fallacy of counting patient OOP as insurer revenue)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    compute_calibration_bins,
    compute_gain_curve,
    evaluate_policies_bootstrap,
    extract_tree_rules,
    fit_model_suite,
    manual_review_analysis,
    subgroup_fairness_audit,
    tune_threshold,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "meps"
OUT_DIR = ROOT / "results" / "exp3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Official BLS CPI-U Medical Care annual averages (for constant 2024 USD normalization)
CPI_MED = {
    2017: 475.3,
    2018: 484.7,
    2019: 498.4,
    2020: 518.9,
    2021: 525.3,
    2022: 546.6,
    2023: 549.1,
    2024: 566.5,
}

PANEL_META = [
    ("h210.dta", 22, 2017, 2018),
    ("h217.dta", 23, 2018, 2019),
    ("h236.dta", 25, 2020, 2021),
    ("h244.dta", 26, 2021, 2022),
    ("h252.dta", 27, 2022, 2023),
    ("h257.dta", 28, 2023, 2024),
]

CHRONIC_COMMON = [
    ("hypertension", "HIBPDXY1"),
    ("chd", "CHDDXY1"),
    ("angina", "ANGIDXY1"),
    ("mi", "MIDXY1"),
    ("other_heart", "OHRTDXY1"),
    ("stroke", "STRKDXY1"),
    ("emphysema", "EMPHDXY1"),
    ("high_chol", "CHOLDXY1"),
    ("arthritis", "ARTHDXY1"),
    ("asthma", "ASTHDXY1"),
    ("cancer", "CANCERY1"),
]


def load_pooled_meps() -> pd.DataFrame:
    frames = []
    for fname, panel_id, y1_yr, y2_yr in PANEL_META:
        df = pd.read_stata(DATA_DIR / fname, convert_categoricals=False)
        mask = (df["INSCOVY1"] == 1) & (df["INSCOVY2"] == 1) & (df["AGEY1X"] >= 18) & (df["AGEY1X"] <= 64)
        sub = df.loc[mask].copy()

        rec = pd.DataFrame(index=sub.index)
        rec["panel"] = panel_id
        rec["y1_year"] = y1_yr
        rec["y2_year"] = y2_yr
        rec["dupersid"] = sub["DUPERSID"].astype(str)
        rec["age"] = sub["AGEY1X"].astype(float)
        rec["sex"] = sub["SEX"].map({1: "Male", 2: "Female"}).fillna("Unknown")
        rec["region"] = sub["REGIONY1"].map({1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}).fillna("Unknown")
        rec["race_eth"] = (
            sub["RACETHX"]
            .map({1: "Hispanic", 2: "NH_White", 3: "NH_Black", 4: "NH_Asian", 5: "NH_Other"})
            .fillna("Unknown")
        )
        rec["marital"] = (
            sub["MARRYY1X"]
            .map({1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never_Married"})
            .fillna("Unknown")
        )
        # Education degree (HIDEG: 1=No deg, 2=GED, 3=HS, 4=Bachelors, 5=Masters, 6=Doc, 7=Other)
        hideg = sub["HIDEG"].where(sub["HIDEG"] > 0, np.nan)
        rec["educ_level"] = hideg.fillna(hideg.median())
        rec["povcat"] = sub["POVCATY1"].where(sub["POVCATY1"] > 0, np.nan).fillna(3).astype(int)
        rec["faminc_y1"] = sub["FAMINCY1"].clip(lower=0).astype(float) * (CPI_MED[2024] / CPI_MED[y1_yr])
        rec["employed_y1"] = (sub["EMPST1"] == 1).astype(float)

        # Health status (1=Excellent ... 5=Poor; negative codes -> median)
        rth = sub["RTHLTH1"].where(sub["RTHLTH1"].between(1, 5), np.nan)
        mnh = sub["MNHLTH1"].where(sub["MNHLTH1"].between(1, 5), np.nan)
        rec["rthlth_y1"] = rth.fillna(rth.median())
        rec["mnhlth_y1"] = mnh.fillna(mnh.median())

        # Smoking (OFTSMK3: 1=Every day, 2=Some days, 3=Not at all)
        smk = sub["OFTSMK3"].where(sub["OFTSMK3"].between(1, 3), np.nan)
        rec["smoker_current"] = smk.isin([1, 2]).astype(float)
        rec["smoker_daily"] = (smk == 1).astype(float)

        # 12 diagnosed chronic conditions (Year 1)
        diab_col = "DIABDXY1_M18" if "DIABDXY1_M18" in sub.columns else "DIABDXY1"
        rec["diabetes"] = (sub[diab_col] == 1).astype(float)
        chron_names = ["diabetes"]
        for cname, raw_col in CHRONIC_COMMON:
            rec[cname] = (sub[raw_col] == 1).astype(float)
            chron_names.append(cname)
        rec["chronic_count"] = rec[chron_names].sum(axis=1)

        # Constant 2024 medical USD expenditures
        f1 = CPI_MED[2024] / CPI_MED[y1_yr]
        f2 = CPI_MED[2024] / CPI_MED[y2_yr]
        rec["totprv_y1"] = sub["TOTPRVY1"].clip(lower=0).astype(float) * f1
        rec["totslf_y1"] = sub["TOTSLFY1"].clip(lower=0).astype(float) * f1
        rec["totexp_y1"] = sub["TOTEXPY1"].clip(lower=0).astype(float) * f1
        rec["had_claim_y1"] = (rec["totprv_y1"] > 0).astype(float)

        rec["totprv_y2"] = sub["TOTPRVY2"].clip(lower=0).astype(float) * f2
        rec["totslf_y2"] = sub["TOTSLFY2"].clip(lower=0).astype(float) * f2
        rec["totexp_y2"] = sub["TOTEXPY2"].clip(lower=0).astype(float) * f2
        frames.append(rec)

    full = pd.concat(frames, ignore_index=True)
    # Age bands for ex-ante actuarial pricing & fairness audit
    full["age_band"] = pd.cut(full["age"], [17, 29, 39, 49, 59, 65], labels=["18-29", "30-39", "40-49", "50-59", "60-64"])
    full["age_group"] = pd.cut(full["age"], [17, 29, 44, 59, 65], labels=["18-29", "30-44", "45-59", "60-64"])
    return full


def compute_exante_margins(df: pd.DataFrame, idx_tr: np.ndarray) -> dict[str, np.ndarray]:
    """Calibrate ex-ante benchmark premiums strictly on training data (zero test leakage)."""
    tr = df.iloc[idx_tr]
    mean_claim_tr = float(tr["totprv_y2"].mean())

    # 1. Community-rated uniform premium targeting 80% MLR
    p_comm_80 = np.full(len(df), mean_claim_tr / 0.80)

    # 2. Age-band actuarial premium calibrated on training set, targeting 80% and 95% MLR
    band_means = tr.groupby("age_band", observed=True)["totprv_y2"].mean().to_dict()
    raw_age_prem = df["age_band"].map(band_means).astype(float).values
    # Normalize so training portfolio MLR equals exact target
    scale_80 = (mean_claim_tr / 0.80) / float(raw_age_prem[idx_tr].mean())
    scale_95 = (mean_claim_tr / 0.95) / float(raw_age_prem[idx_tr].mean())
    p_age_80 = raw_age_prem * scale_80
    p_age_95 = raw_age_prem * scale_95

    claims = df["totprv_y2"].values
    oop = df["totslf_y2"].values

    return {
        "P_comm_80": p_comm_80 - claims,
        "P_age_80": p_age_80 - claims,
        "P_age_95": p_age_95 - claims,
        "P_age_80_D0err": p_age_80 + oop - claims,
    }


def build_meps_features(df: pd.DataFrame, include_prior_cost: bool, drop_sensitive: bool) -> pd.DataFrame:
    clin_cols = [
        "age",
        "chronic_count",
        "diabetes",
        "hypertension",
        "chd",
        "angina",
        "mi",
        "other_heart",
        "stroke",
        "emphysema",
        "high_chol",
        "arthritis",
        "asthma",
        "cancer",
        "smoker_current",
        "smoker_daily",
        "rthlth_y1",
        "mnhlth_y1",
    ]
    if include_prior_cost:
        clin_cols = clin_cols + ["totprv_y1", "totslf_y1", "had_claim_y1"]

    if drop_sensitive:
        return df[clin_cols].astype(float).reset_index(drop=True)

    socio_num = ["educ_level", "povcat", "faminc_y1", "employed_y1"]
    socio_cat = ["sex", "region", "race_eth", "marital"]
    X_cat = pd.get_dummies(df[socio_cat].astype(str), drop_first=True, dtype=float)
    X = pd.concat([df[clin_cols + socio_num].astype(float).reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X


def calibrate_clinical_rule_meps(df: pd.DataFrame, margins: np.ndarray, idx_va: np.ndarray, include_prior_cost: bool) -> tuple[np.ndarray, str]:
    """Calibrate an interpretable clinical safety rule (Recall >= 70%) on validation data."""
    y_va = margins[idx_va] < 0
    m_va = margins[idx_va]
    cc = df["chronic_count"].values
    smk = df["smoker_current"].values > 0
    rth = df["rthlth_y1"].values
    age = df["age"].values
    p1 = df["totprv_y1"].values

    best_flag = None
    best_desc = ""
    best_prof = -np.inf

    if not include_prior_cost:
        for cc_cut in [1, 2, 3, 99]:
            for rth_cut in [3, 4, 5, 99]:
                for age_cut in [40, 45, 50, 55, 60, 99]:
                    for use_smk in [False, True]:
                        flag = (cc >= cc_cut) | (rth >= rth_cut) | (age >= age_cut) | (smk if use_smk else False)
                        rec = (flag[idx_va] & y_va).sum() / max(y_va.sum(), 1)
                        if rec >= 0.70:
                            prof = m_va[~flag[idx_va]].sum()
                            if prof > best_prof:
                                best_prof = prof
                                best_flag = flag
                                best_desc = f"chronic>={cc_cut} OR rthlth>={rth_cut} OR age>={age_cut}" + (" OR smoker" if use_smk else "")
    else:
        for p1_cut in [500, 1000, 2000, 3500, 5000, 10000, 1e9]:
            for cc_cut in [1, 2, 3, 99]:
                for rth_cut in [3, 4, 99]:
                    flag = (p1 >= p1_cut) | (cc >= cc_cut) | (rth >= rth_cut)
                    rec = (flag[idx_va] & y_va).sum() / max(y_va.sum(), 1)
                    if rec >= 0.70:
                        prof = m_va[~flag[idx_va]].sum()
                        if prof > best_prof:
                            best_prof = prof
                            best_flag = flag
                            best_desc = f"totprv_y1>={p1_cut:.0f} OR chronic>={cc_cut} OR rthlth>={rth_cut}"

    if best_flag is None:
        best_flag = np.ones(len(df), dtype=bool)
        best_desc = "refer_all"
    return best_flag, best_desc


def run_meps_setting(
    tag: str,
    df: pd.DataFrame,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    include_prior_cost: bool,
):
    print(f"\n{'='*80}\nRUNNING MEPS SETTING: {tag} (tr={len(idx_tr):,}, va={len(idx_va):,}, te={len(idx_te):,})\n{'='*80}")
    X_full = build_meps_features(df, include_prior_cost=include_prior_cost, drop_sensitive=False)
    X_clin = build_meps_features(df, include_prior_cost=include_prior_cost, drop_sensitive=True)
    margin_dict = compute_exante_margins(df, idx_tr)

    all_proto, all_pairs, all_curves, all_manual, all_fair, all_feat, all_cal = [], [], [], [], [], [], []
    rules_log = []
    df_te_grp = df.iloc[idx_te].copy().reset_index(drop=True)

    for m_name, margins in margin_dict.items():
        m_te = margins[idx_te]
        print(f"\n--- {tag} | {m_name} | Test Accept-All={m_te.sum()/1e6:+.3f}M | Neg%={(m_te<0).mean()*100:.1f}% ---")
        scores, probs, models = fit_model_suite(X_full, margins, idx_tr, random_state=42, include_mlp=True)
        rule_flag, rule_desc = calibrate_clinical_rule_meps(df, margins, idx_va, include_prior_cost=include_prior_cost)
        rules_log.append(f"=== {tag} | {m_name} ===\nSafety Rule (Recall>=70%): {rule_desc}\n")
        for d in (1, 2):
            nm = f"Shallow Tree (depth={d})"
            if nm in models:
                rules_log.append(f"-- {nm} --\n{extract_tree_rules(models[nm], list(X_full.columns))}\n")

        policies_te = {
            "Accept All": np.zeros(len(idx_te), dtype=bool),
            "Oracle": m_te < 0,
            "Rule Engine (Recall>=70%)": rule_flag[idx_te],
        }
        scores_te, probs_te = {}, {}
        for nm, s in scores.items():
            cut = tune_threshold(s, margins, idx_va)
            policies_te[nm] = s[idx_te] >= cut
            scores_te[nm] = s[idx_te]
            if nm in probs:
                probs_te[nm] = probs[nm][idx_te]

        for reg_nm in ["Decision Tree (margin reg)", "HistGradientBoosting (margin reg)", "MLP (margin reg)"]:
            policies_te[f"{reg_nm} (cut=0)"] = scores[reg_nm][idx_te] > 0.0

        pairs = [
            ("HistGradientBoosting (cls)", "Shallow Tree (depth=1)"),
            ("HistGradientBoosting (cls)", "Shallow Tree (depth=2)"),
            ("HistGradientBoosting (cls)", "Decision Tree (depth=5)"),
            ("HistGradientBoosting (cls)", "Logistic Regression"),
            ("HistGradientBoosting (cls)", "MLP (cls)"),
            ("HistGradientBoosting (margin reg)", "HistGradientBoosting (cls)"),
            ("HistGradientBoosting (margin-weighted)", "HistGradientBoosting (cls)"),
            ("Decision Tree (margin-weighted)", "Decision Tree (depth=5)"),
            ("MLP (cls)", "Shallow Tree (depth=2)"),
            ("Decision Tree (depth=5)", "Rule Engine (Recall>=70%)"),
            ("Shallow Tree (depth=2)", "Rule Engine (Recall>=70%)"),
        ]

        df_p, df_pr = evaluate_policies_bootstrap(
            policies_te, m_te, scores_test=scores_te, probs_test=probs_te, paired_comparisons=pairs, n_boot=2000, seed=42, scale_div=1e6
        )
        df_p.insert(0, "margin_def", m_name)
        df_p.insert(0, "setting", tag)
        df_pr.insert(0, "margin_def", m_name)
        df_pr.insert(0, "setting", tag)
        all_proto.append(df_p)
        all_pairs.append(df_pr)
        print(df_p[["policy", "pct_referred", "recall", "precision", "profit", "gain_vs_accept_all", "pct_oracle", "auroc", "brier"]].to_string(index=False))

        for cm in ["Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)", "Decision Tree (depth=5)"]:
            gc = compute_gain_curve(scores_te[cm], m_te)
            gc["setting"] = tag
            gc["margin_def"] = m_name
            gc["model"] = cm
            gc["val_frac_referred"] = float(policies_te[cm].mean())
            all_curves.append(gc)

        for cm in ["Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]:
            cb = compute_calibration_bins(probs_te[cm], m_te, n_bins=10)
            cb["setting"] = tag
            cb["margin_def"] = m_name
            cb["model"] = cm
            all_cal.append(cb)

        am = manual_review_analysis(
            {
                "Rule Engine (Recall>=70%)": policies_te["Rule Engine (Recall>=70%)"],
                "Shallow Tree (depth=2)": policies_te["Shallow Tree (depth=2)"],
                "Decision Tree (depth=5)": policies_te["Decision Tree (depth=5)"],
                "HistGradientBoosting (cls)": policies_te["HistGradientBoosting (cls)"],
                "MLP (cls)": policies_te["MLP (cls)"],
                "Review All": np.ones(len(idx_te), dtype=bool),
            },
            m_te,
            ref_policy="HistGradientBoosting (cls)",
        )
        am.insert(0, "margin_def", m_name)
        am.insert(0, "setting", tag)
        all_manual.append(am)

        if m_name in ("P_age_80", "P_age_95"):
            fa = subgroup_fairness_audit(
                df_te_grp,
                ["sex", "age_group", "race_eth", "povcat", "region"],
                {
                    "Rule Engine (Recall>=70%)": policies_te["Rule Engine (Recall>=70%)"],
                    "Shallow Tree (depth=2)": policies_te["Shallow Tree (depth=2)"],
                    "Decision Tree (depth=5)": policies_te["Decision Tree (depth=5)"],
                    "HistGradientBoosting (cls)": policies_te["HistGradientBoosting (cls)"],
                    "MLP (cls)": policies_te["MLP (cls)"],
                },
                m_te,
            )
            fa.insert(0, "margin_def", m_name)
            fa.insert(0, "setting", tag)
            all_fair.append(fa)

            sc_cl, _, _ = fit_model_suite(X_clin, margins, idx_tr, random_state=42, include_mlp=True)
            for fs_name, sc_d in [("F_full", scores), ("F_clinical_only", sc_cl)]:
                for mnm in ["Shallow Tree (depth=2)", "Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]:
                    cut_f = tune_threshold(sc_d[mnm], margins, idx_va)
                    fl_te = sc_d[mnm][idx_te] >= cut_f
                    all_feat.append(
                        dict(
                            setting=tag,
                            margin_def=m_name,
                            feature_set=fs_name,
                            model=mnm,
                            pct_referred=round(fl_te.mean() * 100.0, 2),
                            profit_M=round(m_te[~fl_te].sum() / 1e6, 4),
                            gain_M=round((m_te[~fl_te].sum() - m_te.sum()) / 1e6, 4),
                        )
                    )

    pd.concat(all_proto, ignore_index=True).to_csv(OUT_DIR / f"{tag}_protocol.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(OUT_DIR / f"{tag}_paired.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(OUT_DIR / f"{tag}_gain_curves.csv", index=False)
    pd.concat(all_manual, ignore_index=True).to_csv(OUT_DIR / f"{tag}_manual_review.csv", index=False)
    pd.concat(all_fair, ignore_index=True).to_csv(OUT_DIR / f"{tag}_fairness.csv", index=False)
    pd.DataFrame(all_feat).to_csv(OUT_DIR / f"{tag}_features.csv", index=False)
    pd.concat(all_cal, ignore_index=True).to_csv(OUT_DIR / f"{tag}_calibration.csv", index=False)
    (OUT_DIR / f"{tag}_rules_text.txt").write_text("\n".join(rules_log), encoding="utf-8")


def run_20_splits_meps(df: pd.DataFrame, include_prior_cost: bool, tag: str):
    print(f"\n--- Running 20 random 60/20/20 splits on MEPS ({tag}) ---")
    X = build_meps_features(df, include_prior_cost=include_prior_cost, drop_sensitive=False)
    rows = []
    for seed in range(20):
        trv, te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed)
        tr, va = train_test_split(trv, test_size=0.25, random_state=seed)
        mdict = compute_exante_margins(df, tr)
        for m_name in ("P_age_80", "P_age_95"):
            m = mdict[m_name]
            base = m[te].sum()
            sc_ = 10_000.0 / len(te) / 1e6  # M USD per 10,000 lives
            scores, _, _ = fit_model_suite(X, m, tr, random_state=seed, include_mlp=True)
            rule_f, _ = calibrate_clinical_rule_meps(df, m, va, include_prior_cost=include_prior_cost)
            r = dict(
                seed=seed,
                margin_def=m_name,
                accept_all_M=base * sc_,
                oracle_gain_M=(m[te][m[te] >= 0].sum() - base) * sc_,
                rule_engine_gain_M=(m[te][~rule_f[te]].sum() - base) * sc_,
            )
            for k, lbl in [
                ("Shallow Tree (depth=1)", "shallow1"),
                ("Shallow Tree (depth=2)", "shallow2"),
                ("Decision Tree (depth=5)", "dt5"),
                ("Decision Tree (margin-weighted)", "dt5_mw"),
                ("Logistic Regression", "logreg"),
                ("HistGradientBoosting (cls)", "hgb_cls"),
                ("HistGradientBoosting (margin reg)", "hgb_reg"),
                ("MLP (cls)", "mlp_cls"),
                ("MLP (margin reg)", "mlp_reg"),
            ]:
                cut = tune_threshold(scores[k], m, va)
                fl = scores[k][te] >= cut
                r[f"{lbl}_gain_M"] = (m[te][~fl].sum() - base) * sc_
                r[f"{lbl}_pct_ref"] = fl.mean() * 100.0
            rows.append(r)

    rp = pd.DataFrame(rows)
    rp.round(4).to_csv(OUT_DIR / f"{tag}_20splits.csv", index=False)
    gain_cols = [c for c in rp.columns if c.endswith("_gain_M") or c == "accept_all_M"]
    print(f"\nMEPS {tag} 20 splits summary (gain in M 2024-USD per 10k lives):")
    print(rp.groupby("margin_def")[gain_cols].agg(["mean", "std"]).round(3).T.to_string())


def main():
    df = load_pooled_meps()
    print(f"Loaded pooled MEPS (Panels 22,23,25,26,27,28): N={len(df):,} continuous-private adults (18-64)")
    print(
        df.groupby("panel")
        .agg(
            n=("age", "size"),
            y1_yr=("y1_year", "first"),
            y2_yr=("y2_year", "first"),
            mean_chronic=("chronic_count", "mean"),
            mean_totprv_y2_2024usd=("totprv_y2", "mean"),
            med_totprv_y2_2024usd=("totprv_y2", "median"),
            mean_totslf_y2_2024usd=("totslf_y2", "mean"),
        )
        .round(2)
        .to_string()
    )
    df.groupby("panel").agg(
        n=("age", "size"),
        y1_year=("y1_year", "first"),
        y2_year=("y2_year", "first"),
        mean_chronic=("chronic_count", "mean"),
        mean_totprv_y2_2024usd=("totprv_y2", "mean"),
        med_totprv_y2_2024usd=("totprv_y2", "median"),
        mean_totslf_y2_2024usd=("totslf_y2", "mean"),
    ).round(2).to_csv(OUT_DIR / "exp3_meps_panel_summary.csv")

    # Strict temporal split: Panels 22-26 (2017-2022) -> train/val, Panels 27-28 (2022-2024) -> test
    idx_trv = np.where(df["panel"].isin([22, 23, 25, 26]))[0]
    idx_te = np.where(df["panel"].isin([27, 28]))[0]
    y_strat = (df["totprv_y2"].values[idx_trv] > df["totprv_y2"].values[idx_trv].mean() / 0.80).astype(int)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=0.25, random_state=42, stratify=y_strat)

    run_meps_setting("exp3a_clinical", df, idx_tr, idx_va, idx_te, include_prior_cost=False)
    run_meps_setting("exp3b_renewal", df, idx_tr, idx_va, idx_te, include_prior_cost=True)

    run_20_splits_meps(df, include_prior_cost=False, tag="exp3a_clinical")
    run_20_splits_meps(df, include_prior_cost=True, tag="exp3b_renewal")


if __name__ == "__main__":
    main()
