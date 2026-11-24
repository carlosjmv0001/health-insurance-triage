#!/usr/bin/env python3
"""Generate publication-grade 300-dpi figures from Exp 1, Exp 2, and Exp 3 CSV results."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

# Colorblind-safe palette (Okabe-Ito inspired)
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def make_fig1_margin_sensitivity():
    """Fig 1: Gain over Accept All vs Fraction Referred across margin definitions in all 3 datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=300)

    # Panel A: MICP (synthetic benchmark)
    ax = axes[0]
    gc1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_gain_curves.csv")
    pr1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_protocol.csv")
    specs1 = [
        ("m_D0", "D0: Deductible as revenue", PAL[0]),
        ("m_D2", "D2: Deductible as cost-sharing", PAL[2]),
        ("m_D1", "D1: Premium minus claims", PAL[1]),
    ]
    for mdef, lbl, col in specs1:
        sub = gc1[(gc1["margin_def"] == mdef) & (gc1["model"] == "HistGradientBoosting (cls)")]
        ax.plot(sub["frac_referred"] * 100, sub["gain"] / 1e6, color=col, lw=2.0, label=lbl)
        vf = float(sub["val_frac_referred"].iloc[0]) * 100
        ax.axvline(vf, color=col, ls=":", lw=1.2, alpha=0.8)
        r_re = pr1[(pr1["margin_def"] == mdef) & (pr1["policy"] == "Rule Engine (Recall>=70%)")]
        if not r_re.empty:
            ax.scatter(
                r_re["pct_referred"],
                r_re["gain_vs_accept_all"],
                color=col,
                marker="X",
                s=55,
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )
    ax.axhline(0, color="black", lw=0.9, alpha=0.6)
    ax.set_title("(a) MICP (Test N = 25,000)")
    ax.set_xlabel("Applicants Referred (%)")
    ax.set_ylabel("Gain over Accept All (M USD)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    # Panel B: SHIP Renewal (Real commercial margin under loading lambda)
    ax = axes[1]
    gc2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_gain_curves.csv")
    pr2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_protocol.csv")
    specs2 = [
        ("M_lam00", r"$\lambda = 0\%$ (Gross, LR 66%)", PAL[0]),
        ("M_lam15", r"$\lambda = 15\%$ (Net, LR 78%)", PAL[2]),
        ("M_lam25", r"$\lambda = 25\%$ (Tight, LR 88%)", PAL[4]),
        ("M_lam35", r"$\lambda = 35\%$ (Deficit, LR 101%)", PAL[1]),
    ]
    for mdef, lbl, col in specs2:
        sub = gc2[(gc2["margin_def"] == mdef) & (gc2["model"] == "HistGradientBoosting (cls)")]
        ax.plot(sub["frac_referred"] * 100, sub["gain"] / 1e6, color=col, lw=2.0, label=lbl)
        vf = float(sub["val_frac_referred"].iloc[0]) * 100
        ax.axvline(vf, color=col, ls=":", lw=1.2, alpha=0.8)
        r_re = pr2[(pr2["margin_def"] == mdef) & (pr2["policy"] == "Rule Engine (Recall>=70%)")]
        if not r_re.empty:
            ax.scatter(
                r_re["pct_referred"],
                r_re["gain_vs_accept_all"],
                color=col,
                marker="X",
                s=55,
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )
    ax.axhline(0, color="black", lw=0.9, alpha=0.6)
    ax.set_title("(b) SHIP Renewal (2019 Test)")
    ax.set_xlabel("Policyholders Referred (%)")
    ax.set_ylabel("Gain over Accept All (M EUR)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    # Panel C: MEPS-HC Clinical Y1 -> Y2 (Real private claims under ex-ante premium rules)
    ax = axes[2]
    gc3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_gain_curves.csv")
    pr3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_protocol.csv")
    specs3 = [
        ("P_comm_80", "Community Premium (80% MLR)", PAL[0]),
        ("P_age_80", "Age-Rated Premium (80% MLR)", PAL[2]),
        ("P_age_95", "Age-Rated Premium (95% MLR)", PAL[1]),
        ("P_age_80_D0err", "Age-Rated + OOP as Rev. (D0 err)", PAL[3]),
    ]
    for mdef, lbl, col in specs3:
        sub = gc3[(gc3["margin_def"] == mdef) & (gc3["model"] == "HistGradientBoosting (cls)")]
        ax.plot(sub["frac_referred"] * 100, sub["gain"] / 1e6, color=col, lw=2.0, label=lbl)
        vf = float(sub["val_frac_referred"].iloc[0]) * 100
        ax.axvline(vf, color=col, ls=":", lw=1.2, alpha=0.8)
        r_re = pr3[(pr3["margin_def"] == mdef) & (pr3["policy"] == "Rule Engine (Recall>=70%)")]
        if not r_re.empty:
            ax.scatter(
                r_re["pct_referred"],
                r_re["gain_vs_accept_all"],
                color=col,
                marker="X",
                s=55,
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )
    ax.axhline(0, color="black", lw=0.9, alpha=0.6)
    ax.set_title("(c) MEPS-HC Clinical Y1→Y2 (2022–2024 Test)")
    ax.set_xlabel("Applicants Referred (%)")
    ax.set_ylabel("Gain over Accept All (M 2024-USD)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG / "fig1_margin_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig2_model_comparison_20splits():
    """Fig 2: Model family comparison across 20 random splits (mean +/- SD) in all 3 datasets."""
    df1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_20splits.csv")
    df2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_20splits.csv")
    df3a = pd.read_csv(RES / "exp3" / "exp3a_clinical_20splits.csv")
    df3b = pd.read_csv(RES / "exp3" / "exp3b_renewal_20splits.csv")

    panels = [
        (df1[df1["margin_def"] == "m_D0"], "(a) MICP D0 (M USD / 25k)", "rule_engine_gain_M", "Rule Engine (Rec≥70%)"),
        (df1[df1["margin_def"] == "m_D2"], "(b) MICP D2 (M USD / 25k)", "rule_engine_gain_M", "Rule Engine (Rec≥70%)"),
        (df2[df2["margin_def"] == "M_lam15"], r"(c) SHIP Renewal $\lambda=15\%$ (M EUR / 50k)", "rule_engine_gain_M", "Rule Engine (Rec≥70%)"),
        (df2[df2["margin_def"] == "M_lam25"], r"(d) SHIP Renewal $\lambda=25\%$ (M EUR / 50k)", "rule_engine_gain_M", "Rule Engine (Rec≥70%)"),
        (df3a[df3a["margin_def"] == "P_age_80"], "(e) MEPS-HC Clinical 80% MLR (M USD / 10k)", "rule_engine_gain_M", "Rule Engine (Rec≥70%)"),
        (df3b[df3b["margin_def"] == "P_age_80"], "(f) MEPS-HC Renewal 80% MLR (M USD / 10k)", "rule_engine_gain_M", "Rule Engine (Rec≥70%)"),
    ]

    model_specs = [
        ("shallow1_gain_M", "Shallow Tree (d=1)", PAL[2]),
        ("shallow2_gain_M", "Shallow Tree (d=2)", PAL[2]),
        ("dt5_gain_M", "Decision Tree (d=5)", PAL[0]),
        ("dt5_mw_gain_M", "Decision Tree (margin-wtd)", PAL[0]),
        ("logreg_gain_M", "Logistic Regression", PAL[5]),
        ("hgb_cls_gain_M", "HistGBM (cls)", PAL[3]),
        ("hgb_reg_gain_M", "HistGBM (margin reg)", PAL[3]),
        ("mlp_cls_gain_M", "MLP (cls)", PAL[4]),
        ("mlp_reg_gain_M", "MLP (margin reg)", PAL[4]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), dpi=300)
    for ax, (sub, title, rule_col, rule_lbl) in zip(axes.ravel(), panels):
        specs = [(rule_col, rule_lbl, PAL[1])] + model_specs
        y_pos = np.arange(len(specs))
        means = [sub[c].mean() for c, _, _ in specs]
        stds = [sub[c].std() for c, _, _ in specs]
        cols = [col for _, _, col in specs]
        lbls = [lbl for _, lbl, _ in specs]

        for yp, mu, sd, col in zip(y_pos, means, stds, cols):
            ax.errorbar(mu, yp, xerr=sd, fmt="o", color=col, ecolor=col, elinewidth=1.8, capsize=3.5, markersize=5.5)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lbls)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Gain over Accept All (mean ± SD, 20 splits)")

    plt.tight_layout()
    fig.savefig(FIG / "fig2_model_comparison_20splits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3_calibration_and_operating_points():
    """Fig 3: Calibration curves (top) and Operating Points in FPR-TPR space (bottom)."""
    c1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_calibration.csv")
    c2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_calibration.csv")
    c3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_calibration.csv")

    p1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_protocol.csv")
    p2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_protocol.csv")
    p3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_protocol.csv")

    cal_panels = [
        (c1[c1["margin_def"] == "m_D0"], "(a) MICP D0 — Calibration"),
        (c2[c2["margin_def"] == "M_lam25"], r"(b) SHIP Renewal $\lambda=25\%$ — Calibration"),
        (c3[c3["margin_def"] == "P_age_80"], "(c) MEPS-HC Clinical 80% MLR — Calibration"),
    ]
    op_panels = [
        (p1[p1["margin_def"] == "m_D0"], "(d) MICP D0 — Operating Points (ROC Space)"),
        (p2[p2["margin_def"] == "M_lam25"], r"(e) SHIP Renewal $\lambda=25\%$ — Operating Points"),
        (p3[p3["margin_def"] == "P_age_80"], "(f) MEPS-HC Clinical 80% MLR — Operating Points"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), dpi=300)
    cal_models = [
        ("Decision Tree (depth=5)", PAL[0], "o"),
        ("Logistic Regression", PAL[2], "s"),
        ("HistGradientBoosting (cls)", PAL[3], "^"),
        ("MLP (cls)", PAL[4], "D"),
    ]
    for ax, (sub, title) in zip(axes[0], cal_panels):
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Perfect calibration")
        for mnm, col, mk in cal_models:
            ms = sub[sub["model"] == mnm]
            if not ms.empty:
                ax.plot(ms["mean_predicted"], ms["fraction_positive"], marker=mk, lw=1.6, ms=4.5, color=col, label=mnm)
        ax.set_title(title)
        ax.set_xlabel("Mean Predicted P(Margin < 0)")
        ax.set_ylabel("Observed Fraction Negative Margin")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=7.5)

    op_models = [
        ("Rule Engine (Recall>=70%)", "Rule Engine (Rec≥70%)", PAL[1], "X", 85),
        ("Shallow Tree (depth=2)", "Shallow Tree (d=2)", PAL[2], "P", 75),
        ("Decision Tree (depth=5)", "Decision Tree (d=5)", PAL[0], "o", 65),
        ("Logistic Regression", "Logistic Regression", PAL[5], "s", 60),
        ("HistGradientBoosting (cls)", "HistGBM (cls)", PAL[3], "^", 70),
        ("MLP (cls)", "MLP (cls)", PAL[4], "D", 60),
    ]
    for ax, (sub, title) in zip(axes[1], op_panels):
        ax.plot([0, 100], [0, 100], "k--", lw=1.0, alpha=0.5)
        seen = set()
        for pol, lbl, col, mk, sz in op_models:
            row = sub[sub["policy"] == pol]
            if not row.empty and lbl not in seen:
                seen.add(lbl)
                ax.scatter(row["fpr"], row["recall"], color=col, marker=mk, s=sz, edgecolor="black", lw=0.5, label=lbl, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate (%)")
        ax.set_ylabel("True Positive Rate / Recall (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.legend(loc="lower right", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(FIG / "fig3_calibration_and_operating_points.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig4_manual_review_sensitivity():
    """Fig 4: Net gain over Accept All under manual review cost c (at q=0.5 and q=1.0)."""
    m1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_manual_review.csv")
    m2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_manual_review.csv")
    m3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_manual_review.csv")

    panels = [
        (m1[m1["margin_def"] == "m_D0"], "(a) MICP D0 (Test N=25k, M USD)", 600, "USD", "Rule Engine (Recall>=70%)"),
        (m2[m2["margin_def"] == "M_lam25"], r"(b) SHIP Renewal $\lambda=25\%$ (M EUR)", 250, "EUR", "Rule Engine (Recall>=70%)"),
        (m3[m3["margin_def"] == "P_age_80"], "(c) MEPS-HC Clinical 80% MLR (M USD)", 3000, "USD", "Rule Engine (Recall>=70%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=300)
    for ax, (sub, title, c_max, curr, rule_nm) in zip(axes, panels):
        c_grid = np.linspace(0, c_max, 100)
        q = 0.50  # realistic 50% human audit efficacy
        specs = [
            (rule_nm, "Rule Engine (Rec≥70%)", PAL[1], "-"),
            ("Shallow Tree (depth=2)", "Shallow Tree (d=2)", PAL[2], "-"),
            ("HistGradientBoosting (cls)", "HistGBM (cls)", PAL[0], "-"),
            ("MLP (cls)", "MLP (cls)", PAL[4], "--"),
            ("Review All", "Review All", "gray", ":"),
        ]
        for pol, lbl, col, ls in specs:
            row = sub[sub["policy"] == pol]
            if not row.empty:
                avoid = float(row["avoidable_loss"].iloc[0])
                n_ref = float(row["n_referred"].iloc[0])
                net_gain = (q * avoid - c_grid * n_ref) / 1e6
                ax.plot(c_grid, net_gain, color=col, ls=ls, lw=1.9, label=lbl)
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(f"Manual Audit Cost per Referred Case, c ({curr})")
        ax.set_ylabel(f"Net Gain over Accept All at q=0.5 (M {curr})")
        ax.legend(loc="upper right", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(FIG / "fig4_manual_review_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig5_fairness_and_ablation():
    """Fig 5: Age-group referral disparity (top) & profit impact of removing sensitive features (bottom)."""
    f1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_fairness.csv")
    f2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_fairness.csv")
    f3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_fairness.csv")

    ab1 = pd.read_csv(RES / "exp1" / "exp1_kaggle_features.csv")
    ab2 = pd.read_csv(RES / "exp2" / "exp2b_renewal_features.csv")
    ab3 = pd.read_csv(RES / "exp3" / "exp3a_clinical_features.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), dpi=300)

    fair_panels = [
        (f1[(f1["margin_def"] == "m_D0") & (f1["group_col"] == "age_group")], "(a) MICP D0 — Referral by Age", "Rule Engine (Recall>=70%)"),
        (f2[(f2["margin_def"] == "M_lam25") & (f2["group_col"] == "age_group")], r"(b) SHIP Renewal $\lambda=25\%$ — Referral by Age", "Rule Engine (Recall>=70%)"),
        (f3[(f3["margin_def"] == "P_age_80") & (f3["group_col"] == "age_group")], "(c) MEPS-HC Clinical 80% MLR — Referral by Age", "Rule Engine (Recall>=70%)"),
    ]
    for ax, (sub, title, r_nm) in zip(axes[0], fair_panels):
        groups = list(dict.fromkeys(sub["group_val"].tolist()))
        x = np.arange(len(groups))
        w = 0.25
        pols = [
            (r_nm, "Rule Engine (Rec≥70%)", PAL[1]),
            ("Shallow Tree (depth=2)", "Shallow Tree (d=2)", PAL[2]),
            ("HistGradientBoosting (cls)", "HistGBM (cls)", PAL[0]),
        ]
        for idx_p, (pol, lbl, col) in enumerate(pols):
            vals = []
            for g in groups:
                r = sub[(sub["group_val"] == g) & (sub["policy"] == pol)]
                vals.append(float(r["pct_referred"].iloc[0]) if not r.empty else 0.0)
            ax.bar(x + (idx_p - 1) * w, vals, width=w, color=col, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 108)
        ax.set_title(title)
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Referral Rate (%)")
        ax.legend(loc="upper left", fontsize=7.5)

    ab_panels = [
        (ab1[ab1["margin_def"] == "m_D0"], "(d) MICP D0 — Feature Ablation", ["F1", "F3", "F2"], ["F1 (Base)", "F3 (No Socio.)", "F2 (+Contract)"], "M USD"),
        (ab2[ab2["margin_def"] == "M_lam25"], r"(e) SHIP Renewal $\lambda=25\%$ — Ablation", ["F_full", "F_no_sensitive"], ["Full (+Geo/Gender)", "No Geo/Gender"], "M EUR"),
        (ab3[ab3["margin_def"] == "P_age_80"], "(f) MEPS-HC Clinical 80% MLR — Ablation", ["F_full", "F_clinical_only"], ["Full (+Socio/Race)", "Clinical Only"], "M 2024-USD"),
    ]
    models_ab = ["Shallow Tree (depth=2)", "Decision Tree (depth=5)", "Logistic Regression", "HistGradientBoosting (cls)", "MLP (cls)"]
    short_m = ["Shallow(d2)", "DT(d5)", "LogReg", "HistGBM", "MLP"]
    for ax, (sub, title, fsets, flbls, unit) in zip(axes[1], ab_panels):
        x = np.arange(len(models_ab))
        w = 0.75 / len(fsets)
        max_v = 0.0
        for idx_f, (fs, flbl) in enumerate(zip(fsets, flbls)):
            vals = []
            for mnm in models_ab:
                r = sub[(sub["feature_set"] == fs) & (sub["model"] == mnm)]
                vals.append(float(r["gain_M"].iloc[0]) if not r.empty else 0.0)
            max_v = max(max_v, max(vals))
            ax.bar(x + (idx_f - (len(fsets) - 1) / 2) * w, vals, width=w, color=PAL[idx_f], label=flbl)
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_ylim(0, max_v * 1.28)
        ax.set_xticks(x)
        ax.set_xticklabels(short_m, rotation=15)
        ax.set_title(title)
        ax.set_ylabel(f"Gain over Accept All ({unit})")
        ax.legend(loc="upper left", ncol=min(len(fsets), 2), fontsize=7.5, frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG / "fig5_fairness_and_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_margin_sensitivity()
    make_fig2_model_comparison_20splits()
    make_fig3_calibration_and_operating_points()
    make_fig4_manual_review_sensitivity()
    make_fig5_fairness_and_ablation()
    print("All 5 figures generated cleanly in:", FIG)
