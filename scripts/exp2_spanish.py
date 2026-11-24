#!/usr/bin/env python3
"""Study 2: SHIP — Spanish Health Insurance Portfolio (Lledo, Espinosa & Perez, Scientific Data, 2026).

Real commercial private health insurer portfolio (2017-2019; 228,711 policy-years).
Excludes Dental-only policies (type_product == 'D', loss ratio 0.16%) to focus on medical coverage.

Two underwriting settings:
  2A. New Business Underwriting (new_business == 'Yes'):
      Train/Val = 2017-2018 (75/25 stratified), Test = 2019 (strict temporal holdout).
  2B. Renewal / Portfolio Triage (paired insured transitions t -> t+1):
      Train/Val = 2017->2018 (75/25), Test = 2018->2019 (strict temporal holdout),
      plus 20 ID-disjoint random partitions (60/20/20) to confirm zero individual memorization.

Margin sensitivity via administrative/acquisition loading lambda in {0%, 15%, 25%, 35%}:
  M(lambda) = (1 - lambda) * premium - cost_claims_year
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
DATA_DIR = ROOT / "data" / "spanish_portfolio"
OUT_DIR = ROOT / "results" / "exp2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDAS = [0.00, 0.15, 0.25, 0.35]

GEO_NUM_COLS = [
    "n_insured_pc",
    "n_insured_mun",
    "n_insured_prov",
    "IICIMUN",
    "IICIPROV",
    "C_GI",
    "C_II",
    "C_IE_P",
    "C_IE_S",
    "C_IE_T",
    "C_GE_P",
    "C_GE_S",
    "C_GE_T",
]
GEO_CAT_COLS = ["C_H", "C_C"]
CONTRACT_CAT_COLS = ["type_policy", "type_policy_dg", "type_product", "reimbursement", "distribution_channel"]


def load_spanish_data() -> pd.DataFrame:
    pkl = DATA_DIR / "spanish_portfolio.pkl"
    if pkl.exists():
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_excel(DATA_DIR / "Dataset of health insurance portfolio.xlsx")
        df.to_pickle(pkl)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    idx_tr: np.ndarray,
) -> pd.DataFrame:
    """One-hot encode categoricals and impute missing numericals using training medians only."""
    X_num = df[num_cols].copy().astype(float)
    meds = X_num.iloc[idx_tr].median()
    X_num = X_num.fillna(meds)

    X_cat = df[cat_cols].copy().astype(str).fillna("MISSING")
    X_dum = pd.get_dummies(X_cat, drop_first=True, dtype=float)
    X = pd.concat([X_num.reset_index(drop=True), X_dum.reset_index(drop=True)], axis=1)
    return X


def calibrate_safety_rule_newbiz(df: pd.DataFrame, margins: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray) -> tuple[np.ndarray, str]:
    """Calibrate a simple operational safety rule (Recall >= 70%) on validation data for new business."""
    y_va = margins[idx_va] < 0
    m_va = margins[idx_va]
    age = df["age"].values
    coll = (df["type_policy"] == "C").values
    reimb = (df["reimbursement"] == "Yes").values

    best_flag = None
    best_desc = ""
    best_prof = -np.inf
    # Search interpretable age thresholds with optional collective/reimbursement triggers
    for age_cut in range(20, 75, 2):
        for desc, flag in [
            (f"age >= {age_cut}", age >= age_cut),
            (f"age >= {age_cut} OR collective", (age >= age_cut) | coll),
            (f"age >= {age_cut} OR reimbursement", (age >= age_cut) | reimb),
            (f"age >= {age_cut} OR collective OR reimbursement", (age >= age_cut) | coll | reimb),
        ]:
            rec = (flag[idx_va] & y_va).sum() / max(y_va.sum(), 1)
            if rec >= 0.70:
                prof = m_va[~flag[idx_va]].sum()
                if prof > best_prof:
                    best_prof = prof
                    best_flag = flag
                    best_desc = desc
    if best_flag is None:
        best_flag = np.ones(len(df), dtype=bool)
        best_desc = "refer_all"
    return best_flag, best_desc


def calibrate_safety_rule_renewal(df: pd.DataFrame, margins: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray) -> tuple[np.ndarray, str]:
    """Calibrate an operational safety rule (Recall >= 70%) on validation data for renewals."""
    y_va = margins[idx_va] < 0
    m_va = margins[idx_va]
    age = df["age"].values
    svc = df["n_medical_services_prev"].values
    clm = df["cost_claims_prev"].values
    lr_p = df["loss_ratio_prev"].values

    best_flag = None
    best_desc = ""
    best_prof = -np.inf
    for svc_cut in [1, 2, 3, 5, 8, 10, 15, 20]:
        for lr_cut in [0.5, 0.75, 1.0, 1.5, 999.0]:
            for age_cut in [45, 55, 60, 65, 999]:
                flag = (svc >= svc_cut) | (lr_p >= lr_cut) | (age >= age_cut)
                rec = (flag[idx_va] & y_va).sum() / max(y_va.sum(), 1)
                if rec >= 0.70:
                    prof = m_va[~flag[idx_va]].sum()
                    if prof > best_prof:
                        best_prof = prof
                        best_flag = flag
                        best_desc = f"svc_prev>={svc_cut} OR lr_prev>={lr_cut} OR age>={age_cut}"
    if best_flag is None:
        best_flag = np.ones(len(df), dtype=bool)
        best_desc = "refer_all"
    return best_flag, best_desc


def run_setting(
    tag: str,
    df: pd.DataFrame,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    num_full: list[str],
    cat_full: list[str],
    num_nosens: list[str],
    cat_nosens: list[str],
    safety_calibrator,
):
    print(f"\n{'='*80}\nRUNNING SPANISH PORTFOLIO SETTING: {tag} (tr={len(idx_tr):,}, va={len(idx_va):,}, te={len(idx_te):,})\n{'='*80}")
    X_full = build_feature_matrix(df, num_full, cat_full, idx_tr)
    X_nosens = build_feature_matrix(df, num_nosens, cat_nosens, idx_tr)

    all_proto = []
    all_pairs = []
    all_curves = []
    all_manual = []
    all_fair = []
    all_feat = []
    all_cal = []
    rules_log = []

    # Prepare demographic subgroup columns on test set
    df_te_grp = df.iloc[idx_te].copy().reset_index(drop=True)
    df_te_grp["age_group"] = pd.cut(df_te_grp["age"], [-1, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"])
    if "C_GI" in df_te_grp.columns:
        df_te_grp["area_income_tertile"] = pd.qcut(
            df_te_grp["C_GI"].fillna(df_te_grp["C_GI"].median()), 3, labels=["Low", "Mid", "High"], duplicates="drop"
        )

    for lam in LAMBDAS:
        m_name = f"M_lam{int(lam*100):02d}"
        margins = (1.0 - lam) * df["premium"].values - df["cost_claims_year"].values
        m_te = margins[idx_te]
        print(f"\n--- {tag} | {m_name} (lambda={lam:.0%}) | Test Accept-All={m_te.sum()/1e6:+.3f}M | Neg%={(m_te<0).mean()*100:.1f}% ---")

        scores, probs, models = fit_model_suite(X_full, margins, idx_tr, random_state=42, include_mlp=True)
        rule_flag, rule_desc = safety_calibrator(df, margins, idx_tr, idx_va)
        rules_log.append(f"=== {tag} | {m_name} ===\nSafety Rule (Recall>=70%): {rule_desc}\n")
        for d in (1, 2):
            nm = f"Shallow Tree (depth={d})"
            if nm in models:
                rules_log.append(f"-- {nm} --\n{extract_tree_rules(models[nm], list(X_full.columns))}\n")

        # Build test-set decisions (threshold tuned strictly on idx_va)
        policies_te: dict[str, np.ndarray] = {
            "Accept All": np.zeros(len(idx_te), dtype=bool),
            "Oracle": m_te < 0,
            "Rule Engine (Recall>=70%)": rule_flag[idx_te],
        }
        scores_te: dict[str, np.ndarray] = {}
        probs_te: dict[str, np.ndarray] = {}
        for nm, s in scores.items():
            cut = tune_threshold(s, margins, idx_va)
            policies_te[nm] = s[idx_te] >= cut
            scores_te[nm] = s[idx_te]
            if nm in probs:
                probs_te[nm] = probs[nm][idx_te]

        # Also evaluate zero-threshold direct margin regression (no cut tuning)
        for reg_nm in ["Decision Tree (margin reg)", "HistGradientBoosting (margin reg)", "MLP (margin reg)"]:
            if reg_nm in scores:
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

        # Gain curves (using HistGBM cls, Logistic, MLP cls, Decision Tree depth=5)
        for curve_model in ["Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)", "Decision Tree (depth=5)"]:
            if curve_model in scores_te:
                gc = compute_gain_curve(scores_te[curve_model], m_te)
                gc["setting"] = tag
                gc["margin_def"] = m_name
                gc["model"] = curve_model
                gc["val_frac_referred"] = float(policies_te[curve_model].mean())
                all_curves.append(gc)

        # Calibration curve bins for calibrated probability models
        for cal_model in ["Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]:
            if cal_model in probs_te:
                cb = compute_calibration_bins(probs_te[cal_model], m_te, n_bins=10)
                cb["setting"] = tag
                cb["margin_def"] = m_name
                cb["model"] = cal_model
                all_cal.append(cb)

        # Manual review sensitivity (c, q)
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

        # Fairness audit on key margins (lambda=15% and 25%)
        if lam in (0.15, 0.25):
            fa = subgroup_fairness_audit(
                df_te_grp,
                ["gender", "age_group", "area_income_tertile", "type_policy"],
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

        # Feature ablation: F_full vs F_no_sensitive (drop gender + all geo/area socioeconomic vars)
        if lam in (0.15, 0.25):
            sc_ns, _, _ = fit_model_suite(X_nosens, margins, idx_tr, random_state=42, include_mlp=True)
            for fs_name, sc_dict in [("F_full", scores), ("F_no_sensitive", sc_ns)]:
                for mnm in ["Shallow Tree (depth=2)", "Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]:
                    cut_f = tune_threshold(sc_dict[mnm], margins, idx_va)
                    fl_te = sc_dict[mnm][idx_te] >= cut_f
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


def run_20_group_splits_renewal(df_ren: pd.DataFrame, num_cols: list[str], cat_cols: list[str]):
    """Run 20 ID-disjoint 60/20/20 random splits on renewal data (lambda=15% and 25%)."""
    print("\n--- Running 20 ID-disjoint random splits on Spanish Renewal (lambda=15%, 25%) ---")
    rows = []
    groups = df_ren["ID"].values
    for seed in range(20):
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        trv, te = next(gss1.split(df_ren, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr_rel, va_rel = next(gss2.split(df_ren.iloc[trv], groups=groups[trv]))
        tr, va = trv[tr_rel], trv[va_rel]

        X = build_feature_matrix(df_ren, num_cols, cat_cols, tr)
        for lam in (0.15, 0.25):
            m = (1.0 - lam) * df_ren["premium"].values - df_ren["cost_claims_year"].values
            base = m[te].sum()
            # Normalize to 50,000 policy-years for clean scale comparison
            sc_ = 50_000.0 / len(te) / 1e6
            scores, _, _ = fit_model_suite(X, m, tr, random_state=seed, include_mlp=True)
            rule_f, _ = calibrate_safety_rule_renewal(df_ren, m, tr, va)

            r = dict(
                seed=seed,
                margin_def=f"M_lam{int(lam*100):02d}",
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
    rp.round(4).to_csv(OUT_DIR / "exp2b_renewal_20splits.csv", index=False)
    gain_cols = [c for c in rp.columns if c.endswith("_gain_M") or c == "accept_all_M"]
    print("\n20 ID-disjoint splits summary (gain in M EUR per 50k policy-years):")
    print(rp.groupby("margin_def")[gain_cols].agg(["mean", "std"]).round(3).T.to_string())


def main():
    df_raw = load_spanish_data()
    # Filter to medical coverage products (exclude Dental-only D with LR 0.16%)
    n_dental = int((df_raw["type_product"] == "D").sum())
    df = df_raw.loc[df_raw["type_product"] != "D"].copy().reset_index(drop=True)
    print(f"Loaded Spanish Portfolio: total={len(df_raw):,} | excluded Dental={n_dental:,} | medical={len(df):,}")

    # ---------------- Setting 2A: New Business Underwriting ----------------
    nb = df.loc[df["new_business"] == "Yes"].copy().reset_index(drop=True)
    # Strict temporal split: 2017-2018 -> train/val, 2019 -> test (drop any ID overlap just in case)
    ids_trv_nb = set(nb.loc[nb["period"].isin([2017, 2018]), "ID"])
    nb = nb.loc[~((nb["period"] == 2019) & (nb["ID"].isin(ids_trv_nb)))].reset_index(drop=True)
    idx_trv_nb = np.where(nb["period"].isin([2017, 2018]))[0]
    idx_te_nb = np.where(nb["period"] == 2019)[0]
    y_strat_nb = (0.85 * nb["premium"].values[idx_trv_nb] - nb["cost_claims_year"].values[idx_trv_nb] < 0).astype(int)
    idx_tr_nb, idx_va_nb = train_test_split(idx_trv_nb, test_size=0.25, random_state=42, stratify=y_strat_nb)

    run_setting(
        tag="exp2a_newbiz",
        df=nb,
        idx_tr=idx_tr_nb,
        idx_va=idx_va_nb,
        idx_te=idx_te_nb,
        num_full=["age"] + GEO_NUM_COLS,
        cat_full=["gender"] + CONTRACT_CAT_COLS + GEO_CAT_COLS,
        num_nosens=["age"],
        cat_nosens=CONTRACT_CAT_COLS,
        safety_calibrator=calibrate_safety_rule_newbiz,
    )

    # ---------------- Setting 2B: Renewal / Portfolio Triage (t -> t+1) ----------------
    prev_cols = ["ID", "period", "cost_claims_year", "n_medical_services", "premium", "exposure_time", "seniority_insured"]
    df_prev = df[prev_cols].copy()
    df_prev["period"] = df_prev["period"] + 1  # align year t features with year t+1 target
    df_prev = df_prev.rename(
        columns={
            "cost_claims_year": "cost_claims_prev",
            "n_medical_services": "n_medical_services_prev",
            "premium": "premium_prev",
            "exposure_time": "exposure_time_prev",
            "seniority_insured": "seniority_prev",
        }
    )
    ren = df.merge(df_prev, on=["ID", "period"], how="inner").reset_index(drop=True)
    ren["loss_ratio_prev"] = ren["cost_claims_prev"] / np.maximum(ren["premium_prev"], 1.0)
    ren["had_claim_prev"] = (ren["cost_claims_prev"] > 0).astype(float)
    print(f"\nPaired renewal transitions: total={len(ren):,} (2017->2018: {(ren['period']==2018).sum():,}, 2018->2019: {(ren['period']==2019).sum():,})")

    idx_trv_r = np.where(ren["period"] == 2018)[0]
    idx_te_r = np.where(ren["period"] == 2019)[0]
    y_strat_r = (0.85 * ren["premium"].values[idx_trv_r] - ren["cost_claims_year"].values[idx_trv_r] < 0).astype(int)
    idx_tr_r, idx_va_r = train_test_split(idx_trv_r, test_size=0.25, random_state=42, stratify=y_strat_r)

    ren_prev_num = ["age", "seniority_prev", "n_medical_services_prev", "cost_claims_prev", "loss_ratio_prev", "exposure_time_prev", "had_claim_prev"]
    run_setting(
        tag="exp2b_renewal",
        df=ren,
        idx_tr=idx_tr_r,
        idx_va=idx_va_r,
        idx_te=idx_te_r,
        num_full=ren_prev_num + GEO_NUM_COLS,
        cat_full=["gender"] + CONTRACT_CAT_COLS + GEO_CAT_COLS,
        num_nosens=ren_prev_num,
        cat_nosens=CONTRACT_CAT_COLS,
        safety_calibrator=calibrate_safety_rule_renewal,
    )

    run_20_group_splits_renewal(ren, num_cols=ren_prev_num + GEO_NUM_COLS, cat_cols=["gender"] + CONTRACT_CAT_COLS + GEO_CAT_COLS)


if __name__ == "__main__":
    main()
