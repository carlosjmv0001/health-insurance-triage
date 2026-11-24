#!/usr/bin/env python3
"""Study 1: Synthetic Benchmark Audit — MICP (Medical Insurance Cost Prediction, Kaggle; 100k x 54).

Evaluates profit-driven underwriting triage under full model parity across the study:
- Three margin definitions:
    D0 = annual_premium + deductible + copay * claims_count - total_claims_paid  (deductible as revenue)
    D2 = annual_premium - max(0, total_claims_paid - deductible - copay * claims_count)  (deductible as cost-sharing)
    D1 = annual_premium - total_claims_paid  (premium minus claims)
- Unified model suite (Shallow Trees depth 1 & 2, Decision Tree depth 5, Margin-Weighted & Margin-Regression
  Trees, Logistic Regression, HistGBM cls/weighted/reg, MLP cls/reg) against Accept All, Oracle, and a
  six-rule clinical Rule Engine targeting Recall >= 70%
- Paired bootstrap (2,000 resamples), 20 random 60/20/20 partitions, feature sets F1/F2/F3,
  rule-by-rule breakdown, manual review (c, q) sensitivity, calibration, and subgroup fairness audit.
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
CSV_PATH = ROOT / "data" / "kaggle" / "medical_insurance.csv"
OUT_DIR = ROOT / "results" / "exp1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAK_COLS = [
    "visits_last_year",
    "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs",
    "medication_count",
    "proc_imaging_count",
    "proc_surgery_count",
    "proc_physio_count",
    "proc_consult_count",
    "proc_lab_count",
    "had_major_procedure",
    "network_tier",
    "deductible",
    "copay",
    "policy_term_years",
    "policy_changes_last_2yrs",
    "provider_quality",
    "annual_premium",
    "monthly_premium",
    "claims_count",
    "avg_claim_amount",
    "total_claims_paid",
    "annual_medical_cost",
    "risk_score",
    "is_high_risk",
]


def prepare_kaggle_features(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prep = raw.copy()
    edu_order = ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"]
    prep["education_num"] = pd.Categorical(prep["education"], categories=edu_order, ordered=True).codes
    X = prep.drop(columns=LEAK_COLS + ["person_id", "education", "household_size"])
    X["sex"] = X["sex"].replace("Other", X["sex"].mode()[0])
    F1 = pd.get_dummies(X, columns=X.select_dtypes(include=["object", "string"]).columns, drop_first=True).astype(float)

    # F2 adds legitimate ex-ante quotation contract variables (deductible, copay, network_tier)
    F2 = F1.copy()
    F2["deductible"] = raw["deductible"].astype(float)
    F2["copay"] = raw["copay"].astype(float)
    F2 = pd.concat([F2, pd.get_dummies(raw["network_tier"], prefix="tier", drop_first=True).astype(float)], axis=1)

    # F3 removes all sociodemographic and plan-type variables
    socio = [
        c
        for c in F1.columns
        if c.split("_")[0] in ("sex", "region", "urban", "marital", "employment", "plan")
        or c in ("income", "education_num", "dependents")
    ]
    F3 = F1.drop(columns=socio)
    return {"F1": F1, "F2": F2, "F3": F3}


def build_clinical_rule_engine(df: pd.DataFrame):
    """Six-rule clinical heuristic targeting Recall >= 70% under D0."""
    cc = df["chronic_count"].values
    cur = (df["smoker"] == "Current").values
    frm = (df["smoker"] == "Former").values
    rule_conds = [
        ("Polymorbidity: chronic >= 2", cc >= 2),
        ("1 chronic & (age > 45 OR SBP > 125)", (cc == 1) & ((df["age"] > 45) | (df["systolic_bp"] > 125)).values),
        ("0 chronic: SBP > 130 & LDL > 130", (cc == 0) & ((df["systolic_bp"] > 130) & (df["ldl"] > 130)).values),
        ("0 chronic: current smoker", (cc == 0) & cur),
        ("0 chronic: former smoker & BMI > 30", (cc == 0) & frm & (df["bmi"] > 30).values),
        ("0 chronic: age > 60", (cc == 0) & (df["age"] > 60).values),
    ]
    rule_engine = np.any([c for _, c in rule_conds], axis=0)
    return rule_engine, rule_conds


def main():
    raw = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[])
    df = raw.copy()
    df["m_D0"] = df["annual_premium"] + df["deductible"] + df["copay"] * df["claims_count"] - df["total_claims_paid"]
    df["m_D2"] = df["annual_premium"] - np.maximum(0.0, df["total_claims_paid"] - df["deductible"] - df["copay"] * df["claims_count"])
    df["m_D1"] = df["annual_premium"] - df["total_claims_paid"]
    df["risk_flag"] = (df["m_D0"] < 0).astype(int)

    FS = prepare_kaggle_features(raw)
    rule_engine, rule_conds = build_clinical_rule_engine(df)

    # Stratified split: 56.25% train / 18.75% validation / 25% held-out test
    y = df["risk_flag"].values
    idx_trv, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=y)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=0.25, random_state=42, stratify=y[idx_trv])

    # Dataset accounting anomalies table (by deductible)
    g = (
        df.groupby("deductible")
        .agg(
            n=("risk_flag", "size"),
            pct_high_risk=("risk_flag", lambda s: s.mean() * 100),
            mean_premium=("annual_premium", "mean"),
            mean_claims=("total_claims_paid", "mean"),
            mean_chronic=("chronic_count", "mean"),
            mean_m_D0=("m_D0", "mean"),
            mean_m_D2=("m_D2", "mean"),
        )
        .round(2)
    )
    g.to_csv(OUT_DIR / "exp1_dataset_structure.csv")

    # Exclusive rule-by-rule breakdown on test set
    done = np.zeros(len(idx_te), dtype=bool)
    rule_rows = []
    for nm, cond in rule_conds + [("Fast-track (no rule triggered)", np.ones(len(df), dtype=bool))]:
        c = cond[idx_te] & ~done
        done |= c
        rule_rows.append(
            dict(
                rule=nm,
                n=int(c.sum()),
                pct_neg_D0=round((df["m_D0"].values[idx_te][c] < 0).mean() * 100, 1),
                total_m_D0_M=round(df["m_D0"].values[idx_te][c].sum() / 1e6, 3),
                mean_m_D0=round(df["m_D0"].values[idx_te][c].mean(), 1),
                total_m_D2_M=round(df["m_D2"].values[idx_te][c].sum() / 1e6, 3),
                mean_m_D2=round(df["m_D2"].values[idx_te][c].mean(), 1),
            )
        )
    pd.DataFrame(rule_rows).to_csv(OUT_DIR / "exp1_rule_breakdown.csv", index=False)

    # Main protocol across D0, D2, D1
    all_proto, all_pairs, all_curves, all_manual, all_fair, all_feat, all_cal = [], [], [], [], [], [], []
    rules_log = []

    df_te_grp = df.iloc[idx_te].copy().reset_index(drop=True)
    df_te_grp["income_tertile"] = pd.qcut(df_te_grp["income"], 3, labels=["Low", "Mid", "High"])
    df_te_grp["age_group"] = pd.cut(df_te_grp["age"], [-1, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"])

    for mcol in ["m_D0", "m_D2", "m_D1"]:
        m = df[mcol].values
        m_te = m[idx_te]
        print(f"\n=== KAGGLE | {mcol} | Test Accept-All={m_te.sum()/1e6:+.3f}M | Neg%={(m_te<0).mean()*100:.1f}% ===")
        scores, probs, models = fit_model_suite(FS["F1"], m, idx_tr, random_state=42, include_mlp=True)
        for d in (1, 2):
            rules_log.append(f"=== {mcol} | Shallow Tree (depth={d}) ===\n{extract_tree_rules(models[f'Shallow Tree (depth={d})'], list(FS['F1'].columns))}\n")

        policies_te = {
            "Accept All": np.zeros(len(idx_te), dtype=bool),
            "Oracle": m_te < 0,
            "Rule Engine (Recall>=70%)": rule_engine[idx_te],
        }
        scores_te, probs_te = {}, {}
        for nm, s in scores.items():
            cut = tune_threshold(s, m, idx_va)
            policies_te[nm] = s[idx_te] >= cut
            scores_te[nm] = s[idx_te]
            if nm in probs:
                probs_te[nm] = probs[nm][idx_te]

        for reg_nm in ["Decision Tree (margin reg)", "HistGradientBoosting (margin reg)", "MLP (margin reg)"]:
            policies_te[f"{reg_nm} (cut=0)"] = scores[reg_nm][idx_te] > 0.0

        pairs = [
            ("Decision Tree (depth=5)", "Rule Engine (Recall>=70%)"),
            ("HistGradientBoosting (cls)", "Shallow Tree (depth=1)"),
            ("HistGradientBoosting (cls)", "Shallow Tree (depth=2)"),
            ("HistGradientBoosting (cls)", "Decision Tree (depth=5)"),
            ("HistGradientBoosting (cls)", "Logistic Regression"),
            ("HistGradientBoosting (cls)", "MLP (cls)"),
            ("MLP (cls)", "Shallow Tree (depth=2)"),
            ("Decision Tree (margin-weighted)", "Decision Tree (depth=5)"),
            ("HistGradientBoosting (margin reg)", "HistGradientBoosting (cls)"),
        ]

        df_p, df_pr = evaluate_policies_bootstrap(
            policies_te, m_te, scores_test=scores_te, probs_test=probs_te, paired_comparisons=pairs, n_boot=2000, seed=42, scale_div=1e6
        )
        df_p.insert(0, "margin_def", mcol)
        df_pr.insert(0, "margin_def", mcol)
        all_proto.append(df_p)
        all_pairs.append(df_pr)
        print(df_p[["policy", "pct_referred", "recall", "precision", "profit", "gain_vs_accept_all", "pct_oracle", "auroc", "brier"]].to_string(index=False))

        for cm in ["Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)", "Decision Tree (depth=5)"]:
            gc = compute_gain_curve(scores_te[cm], m_te)
            gc["margin_def"] = mcol
            gc["model"] = cm
            gc["val_frac_referred"] = float(policies_te[cm].mean())
            all_curves.append(gc)

        for cm in ["Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]:
            cb = compute_calibration_bins(probs_te[cm], m_te, n_bins=10)
            cb["margin_def"] = mcol
            cb["model"] = cm
            all_cal.append(cb)

        if mcol in ("m_D0", "m_D2"):
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
                ref_policy="Decision Tree (depth=5)",
            )
            am.insert(0, "margin_def", mcol)
            all_manual.append(am)

            # Feature ablation F1, F2 (+contractual), F3 (-sociodemographic)
            for fs_name in ["F1", "F2", "F3"]:
                sc_f = scores if fs_name == "F1" else fit_model_suite(FS[fs_name], m, idx_tr, random_state=42, include_mlp=True)[0]
                for mnm in ["Shallow Tree (depth=2)", "Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]:
                    cut_f = tune_threshold(sc_f[mnm], m, idx_va)
                    fl_te = sc_f[mnm][idx_te] >= cut_f
                    all_feat.append(
                        dict(
                            margin_def=mcol,
                            feature_set=fs_name,
                            model=mnm,
                            pct_referred=round(fl_te.mean() * 100.0, 2),
                            profit_M=round(m_te[~fl_te].sum() / 1e6, 4),
                            gain_M=round((m_te[~fl_te].sum() - m_te.sum()) / 1e6, 4),
                        )
                    )

        if mcol == "m_D0":
            fa = subgroup_fairness_audit(
                df_te_grp,
                ["sex", "income_tertile", "region", "urban_rural", "age_group"],
                {
                    "Rule Engine (Recall>=70%)": policies_te["Rule Engine (Recall>=70%)"],
                    "Shallow Tree (depth=2)": policies_te["Shallow Tree (depth=2)"],
                    "Decision Tree (depth=5)": policies_te["Decision Tree (depth=5)"],
                    "HistGradientBoosting (cls)": policies_te["HistGradientBoosting (cls)"],
                    "MLP (cls)": policies_te["MLP (cls)"],
                },
                m_te,
            )
            fa.insert(0, "margin_def", mcol)
            all_fair.append(fa)

    pd.concat(all_proto, ignore_index=True).to_csv(OUT_DIR / "exp1_kaggle_protocol.csv", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(OUT_DIR / "exp1_kaggle_paired.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(OUT_DIR / "exp1_kaggle_gain_curves.csv", index=False)
    pd.concat(all_manual, ignore_index=True).to_csv(OUT_DIR / "exp1_kaggle_manual_review.csv", index=False)
    pd.concat(all_fair, ignore_index=True).to_csv(OUT_DIR / "exp1_kaggle_fairness.csv", index=False)
    pd.DataFrame(all_feat).to_csv(OUT_DIR / "exp1_kaggle_features.csv", index=False)
    pd.concat(all_cal, ignore_index=True).to_csv(OUT_DIR / "exp1_kaggle_calibration.csv", index=False)
    (OUT_DIR / "exp1_kaggle_rules_text.txt").write_text("\n".join(rules_log), encoding="utf-8")

    # 20 random 60/20/20 splits (D0 & D2)
    print("\n--- Running 20 random 60/20/20 splits on Kaggle (D0 & D2) ---")
    s_rows = []
    for seed in range(20):
        trv, te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed, stratify=y)
        tr, va = train_test_split(trv, test_size=0.25, random_state=seed, stratify=y[trv])
        for mcol in ("m_D0", "m_D2"):
            m = df[mcol].values
            base = m[te].sum()
            sc_ = 25_000.0 / len(te) / 1e6
            scores, _, _ = fit_model_suite(FS["F1"], m, tr, random_state=seed, include_mlp=True)
            r = dict(
                seed=seed,
                margin_def=mcol,
                accept_all_M=base * sc_,
                oracle_gain_M=(m[te][m[te] >= 0].sum() - base) * sc_,
                rule_engine_gain_M=(m[te][~rule_engine[te]].sum() - base) * sc_,
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
            s_rows.append(r)

    rp = pd.DataFrame(s_rows)
    rp.round(4).to_csv(OUT_DIR / "exp1_kaggle_20splits.csv", index=False)
    gain_cols = [c for c in rp.columns if c.endswith("_gain_M") or c == "accept_all_M"]
    print("\nKaggle 20 splits summary (gain in M USD per 25k lives):")
    print(rp.groupby("margin_def")[gain_cols].agg(["mean", "std"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
