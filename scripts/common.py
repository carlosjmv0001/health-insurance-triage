#!/usr/bin/env python3
"""Unified experimental evaluation engine for profit-driven underwriting triage.

Enforces methodological consistency across all studies:
- Threshold selection strictly on validation split (never on test)
- Single decision operator (score >= cut) for both profit and confusion metrics
- Paired bootstrap (2,000 resamples) for 95% CIs on profits, gains, and model differences
- Calibration (Brier score + reliability curve bins) and probability unbalancing for class-weighted trees
- Manual underwriting review sensitivity (audit cost c, detection efficacy q)
- Subgroup fairness audit (referral rate, FPR, TPR, precision)
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def unbalance_proba(p_bal: np.ndarray, pos_prior: float) -> np.ndarray:
    """Convert class_weight='balanced' probabilities back to true empirical probabilities."""
    p = np.clip(p_bal, 1e-9, 1.0 - 1e-9)
    ratio = (1.0 - pos_prior) / max(pos_prior, 1e-9)
    odds_true = (p / (1.0 - p)) / ratio
    return odds_true / (1.0 + odds_true)


def tune_threshold(
    scores: np.ndarray,
    margins: np.ndarray,
    idx_val: np.ndarray,
    n_quantiles: int = 401,
) -> float:
    """Select decision threshold on validation data maximizing realized profit.

    Decision rule: refer (remove, profit=0) if score >= cut; accept otherwise.
    Ties break in favor of higher threshold (fewer referrals).
    """
    s_v = np.asarray(scores[idx_val], dtype=float)
    m_v = np.asarray(margins[idx_val], dtype=float)
    uniq = np.unique(s_v)
    if len(uniq) <= 300:
        cands = uniq
    else:
        cands = np.unique(np.quantile(s_v, np.linspace(0.0, 1.0, n_quantiles)))
    # Include accept-all (+inf) and refer-all (-inf)
    cands = np.concatenate([[-np.inf], cands, [np.inf]])

    # Vectorized or fast sorted evaluation
    order = np.argsort(s_v)
    s_ord = s_v[order]
    m_ord = m_v[order]
    csum = np.concatenate([[0.0], np.cumsum(m_ord)])

    # For score < c, index is searchsorted(s_ord, c, side='left')
    pos = np.searchsorted(s_ord, cands, side="left")
    profits = csum[pos]

    # Prefer higher cut (fewer referrals) on exact ties
    best_profit = profits.max()
    best_indices = np.where(np.isclose(profits, best_profit, rtol=0.0, atol=1e-9))[0]
    return float(cands[best_indices[-1]])


def fit_model_suite(
    X_all: pd.DataFrame,
    margins_all: np.ndarray,
    idx_tr: np.ndarray,
    random_state: int = 42,
    include_mlp: bool = True,
    min_leaf_deep: int = 25,
    min_leaf_shallow: int = 50,
    dt_orig_min_leaf: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Fit all model families on idx_tr and score all rows in X_all.

    Returns:
      scores: dict[model_name -> risk score array] (higher = riskier / more likely to refer)
      probs:  dict[model_name -> calibrated P(margin < 0) array] (for probability/Brier analysis)
      models: dict[model_name -> fitted estimator]
    """
    X_tr = X_all.iloc[idx_tr]
    m_tr = np.asarray(margins_all[idx_tr], dtype=float)
    y_tr = (m_tr < 0).astype(int)
    pos_prior = float(y_tr.mean())
    abs_w = np.maximum(np.abs(m_tr), 1e-6)

    scores: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    models: dict[str, object] = {}

    # Guard against degenerate single-class targets
    if len(np.unique(y_tr)) < 2:
        const_s = np.full(len(X_all), float(y_tr[0]))
        for name in [
            "Shallow Tree (depth=1)",
            "Shallow Tree (depth=2)",
            "Decision Tree (depth=5)",
            "Decision Tree (margin-weighted)",
            "Decision Tree (margin reg)",
            "Logistic Regression",
            "HistGradientBoosting (cls)",
            "HistGradientBoosting (margin-weighted)",
            "HistGradientBoosting (margin reg)",
        ]:
            scores[name] = const_s.copy()
        if include_mlp:
            scores["MLP (cls)"] = const_s.copy()
            scores["MLP (margin reg)"] = const_s.copy()
        return scores, probs, models

    # 1-2. Interpretable-by-design shallow trees (automatic rule induction, zero data snooping)
    for d in (1, 2):
        name = f"Shallow Tree (depth={d})"
        clf = DecisionTreeClassifier(
            max_depth=d, min_samples_leaf=min_leaf_shallow, class_weight="balanced", random_state=random_state
        ).fit(X_tr, y_tr)
        p_bal = clf.predict_proba(X_all)[:, 1]
        scores[name] = p_bal
        probs[name] = unbalance_proba(p_bal, pos_prior)
        models[name] = clf

    # 3. Decision Tree (depth=5)
    leaf5 = dt_orig_min_leaf if dt_orig_min_leaf is not None else min_leaf_deep
    dt5 = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=leaf5, class_weight="balanced", random_state=random_state
    ).fit(X_tr, y_tr)
    p_dt5 = dt5.predict_proba(X_all)[:, 1]
    scores["Decision Tree (depth=5)"] = p_dt5
    probs["Decision Tree (depth=5)"] = unbalance_proba(p_dt5, pos_prior)
    models["Decision Tree (depth=5)"] = dt5

    # 4. Margin-weighted Decision Tree (example-dependent cost-sensitive proxy)
    dt_w = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=min_leaf_deep, random_state=random_state
    ).fit(X_tr, y_tr, sample_weight=abs_w)
    scores["Decision Tree (margin-weighted)"] = dt_w.predict_proba(X_all)[:, 1]
    models["Decision Tree (margin-weighted)"] = dt_w

    # 5. Direct margin regression Decision Tree
    dt_reg = DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=min_leaf_deep, random_state=random_state
    ).fit(X_tr, m_tr)
    scores["Decision Tree (margin reg)"] = -dt_reg.predict(X_all)
    models["Decision Tree (margin reg)"] = dt_reg

    # 6. Logistic Regression (unweighted -> naturally calibrated probabilities)
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
    ).fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_all)[:, 1]
    scores["Logistic Regression"] = p_lr
    probs["Logistic Regression"] = p_lr
    models["Logistic Regression"] = lr

    # 7. HistGradientBoosting (classification)
    hgb = HistGradientBoostingClassifier(
        random_state=random_state, early_stopping=True, max_iter=200
    ).fit(X_tr, y_tr)
    p_hgb = hgb.predict_proba(X_all)[:, 1]
    scores["HistGradientBoosting (cls)"] = p_hgb
    probs["HistGradientBoosting (cls)"] = p_hgb
    models["HistGradientBoosting (cls)"] = hgb

    # 8. HistGradientBoosting (margin-weighted)
    hgb_w = HistGradientBoostingClassifier(
        random_state=random_state, early_stopping=True, max_iter=200
    ).fit(X_tr, y_tr, sample_weight=abs_w)
    scores["HistGradientBoosting (margin-weighted)"] = hgb_w.predict_proba(X_all)[:, 1]
    models["HistGradientBoosting (margin-weighted)"] = hgb_w

    # 9. HistGradientBoosting (direct margin regression)
    hgb_reg = HistGradientBoostingRegressor(
        random_state=random_state, early_stopping=True, max_iter=200
    ).fit(X_tr, m_tr)
    scores["HistGradientBoosting (margin reg)"] = -hgb_reg.predict(X_all)
    models["HistGradientBoosting (margin reg)"] = hgb_reg

    # 10-11. Neural Network (MLP) — classification & direct margin regression
    if include_mlp:
        mlp_cls = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=80,
                early_stopping=True,
                n_iter_no_change=8,
                batch_size=min(256, max(32, len(X_tr) // 20)),
                random_state=random_state,
            ),
        ).fit(X_tr, y_tr)
        p_mlp = mlp_cls.predict_proba(X_all)[:, 1]
        scores["MLP (cls)"] = p_mlp
        probs["MLP (cls)"] = p_mlp
        models["MLP (cls)"] = mlp_cls

        # Scale target margins for stable MLP regression training
        m_std = float(np.std(m_tr)) if np.std(m_tr) > 1e-6 else 1.0
        m_mean = float(np.mean(m_tr))
        mlp_reg = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=80,
                early_stopping=True,
                n_iter_no_change=8,
                batch_size=min(256, max(32, len(X_tr) // 20)),
                random_state=random_state,
            ),
        ).fit(X_tr, (m_tr - m_mean) / m_std)
        pred_m = mlp_reg.predict(X_all) * m_std + m_mean
        scores["MLP (margin reg)"] = -pred_m
        models["MLP (margin reg)"] = mlp_reg

    return scores, probs, models


def evaluate_policies_bootstrap(
    policies_test: dict[str, np.ndarray],
    margins_test: np.ndarray,
    scores_test: dict[str, np.ndarray] | None = None,
    probs_test: dict[str, np.ndarray] | None = None,
    paired_comparisons: list[tuple[str, str]] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    scale_div: float = 1e6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all policies on test set with paired bootstrap 95% CIs."""
    m = np.asarray(margins_test, dtype=float)
    y = m < 0
    n = len(m)
    names = list(policies_test.keys())
    if "Accept All" not in policies_test:
        raise ValueError("policies_test must include 'Accept All' as first baseline.")
    acc_idx = names.index("Accept All")

    # Profit contribution matrix K[j, i] = 0 if referred else m[i]
    K = np.array([np.where(np.asarray(policies_test[nm], dtype=bool), 0.0, m) for nm in names])

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(names)), dtype=float)
    # Chunked bootstrap for speed & low memory
    chunk = 200
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        b_sz = end - start
        # Multinomial resampling weights equivalent to sampling with replacement
        idx_mat = rng.integers(0, n, size=(b_sz, n))
        for b in range(b_sz):
            boots[start + b] = K[:, idx_mat[b]].sum(axis=1)

    oracle_gain = float((m[m >= 0].sum() - m.sum()))
    acc_profit = float(K[acc_idx].sum())

    rows = []
    for j, nm in enumerate(names):
        f = np.asarray(policies_test[nm], dtype=bool)
        tp = int((f & y).sum())
        fp = int((f & ~y).sum())
        fn = int((~f & y).sum())
        tn = int((~f & ~y).sum())
        n_ref = int(f.sum())
        recall = tp / max(int(y.sum()), 1)
        prec = tp / max(n_ref, 1)
        fpr = fp / max(int((~y).sum()), 1)
        acc = (tp + tn) / n
        f1 = 2 * prec * recall / max(prec + recall, 1e-12)

        prof = float(K[j].sum())
        gain = prof - acc_profit
        d_boot = boots[:, j] - boots[:, acc_idx]
        pct_oracle = (gain / oracle_gain * 100.0) if oracle_gain > 1e-9 else np.nan

        auroc = np.nan
        aupr = np.nan
        brier = np.nan
        if scores_test and nm in scores_test and len(np.unique(y)) == 2:
            auroc = float(roc_auc_score(y, scores_test[nm]))
            aupr = float(average_precision_score(y, scores_test[nm]))
        if probs_test and nm in probs_test and len(np.unique(y)) == 2:
            brier = float(brier_score_loss(y, np.clip(probs_test[nm], 0.0, 1.0)))

        rows.append(
            dict(
                policy=nm,
                n_referred=n_ref,
                pct_referred=round(f.mean() * 100.0, 2),
                recall=round(recall * 100.0, 2),
                precision=round(prec * 100.0, 2),
                fpr=round(fpr * 100.0, 2),
                accuracy=round(acc * 100.0, 2),
                f1=round(f1 * 100.0, 2),
                profit=round(prof / scale_div, 4),
                profit_ci_lo=round(float(np.percentile(boots[:, j], 2.5)) / scale_div, 4),
                profit_ci_hi=round(float(np.percentile(boots[:, j], 97.5)) / scale_div, 4),
                gain_vs_accept_all=round(gain / scale_div, 4),
                gain_ci_lo=round(float(np.percentile(d_boot, 2.5)) / scale_div, 4),
                gain_ci_hi=round(float(np.percentile(d_boot, 97.5)) / scale_div, 4),
                pct_oracle=round(pct_oracle, 2) if not np.isnan(pct_oracle) else np.nan,
                auroc=round(auroc, 4) if not np.isnan(auroc) else np.nan,
                aupr=round(aupr, 4) if not np.isnan(aupr) else np.nan,
                brier=round(brier, 4) if not np.isnan(brier) else np.nan,
            )
        )

    pair_rows = []
    if paired_comparisons:
        for a, b in paired_comparisons:
            if a in names and b in names:
                ia, ib = names.index(a), names.index(b)
                diff = float(K[ia].sum() - K[ib].sum())
                db = boots[:, ia] - boots[:, ib]
                p_val = float(min(1.0, 2.0 * min((db <= 0).mean(), (db >= 0).mean())))
                pair_rows.append(
                    dict(
                        policy_a=a,
                        policy_b=b,
                        diff=round(diff / scale_div, 4),
                        ci_lo=round(float(np.percentile(db, 2.5)) / scale_div, 4),
                        ci_hi=round(float(np.percentile(db, 97.5)) / scale_div, 4),
                        p_boot=round(p_val, 4),
                    )
                )

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def compute_gain_curve(scores_test: np.ndarray, margins_test: np.ndarray, n_grid: int = 201) -> pd.DataFrame:
    """Compute gain over Accept All as a function of fraction referred (ordered by descending score)."""
    s = np.asarray(scores_test, dtype=float)
    m = np.asarray(margins_test, dtype=float)
    # Break ties deterministically with small stable order
    order = np.argsort(-s, kind="mergesort")
    m_ord = m[order]
    n = len(m)
    # Removing first k riskiest individuals yields gain = -sum(m_ord[:k])
    cum_gain = np.concatenate([[0.0], -np.cumsum(m_ord)])
    ks = np.unique(np.round(np.linspace(0, n, n_grid)).astype(int))
    return pd.DataFrame(
        {
            "frac_referred": ks / n,
            "n_referred": ks,
            "gain": cum_gain[ks],
            "profit": m.sum() + cum_gain[ks],
        }
    )


def manual_review_analysis(
    policies_test: dict[str, np.ndarray],
    margins_test: np.ndarray,
    ref_policy: str,
) -> pd.DataFrame:
    """Two-parameter manual underwriting review model (audit cost c, detection efficacy q).

    When a referred applicant is reviewed (cost c) instead of automatically rejected,
    a fraction q of truly negative-margin applicants is rejected/re-priced to margin 0,
    while profitable referred applicants are retained.
    Net gain over Accept All = q * avoidable_loss - c * n_referred.
    """
    m = np.asarray(margins_test, dtype=float)
    rows = []
    for nm, f in policies_test.items():
        fb = np.asarray(f, dtype=bool)
        n_ref = int(fb.sum())
        avoidable = float(-m[fb & (m < 0)].sum())
        c_be_acc = (avoidable / n_ref) if n_ref > 0 else np.nan
        rows.append(
            dict(
                policy=nm,
                n_referred=n_ref,
                pct_referred=round(fb.mean() * 100.0, 2),
                avoidable_loss=round(avoidable, 2),
                breakeven_c_vs_accept_all_q1=round(c_be_acc, 2) if not np.isnan(c_be_acc) else np.nan,
            )
        )
    df_am = pd.DataFrame(rows)
    if ref_policy in df_am["policy"].values:
        ref = df_am.set_index("policy").loc[ref_policy]
        c_vs_ref = []
        for _, r in df_am.iterrows():
            dn = r["n_referred"] - ref["n_referred"]
            if dn != 0:
                c_vs_ref.append(round((r["avoidable_loss"] - ref["avoidable_loss"]) / dn, 2))
            else:
                c_vs_ref.append(np.nan)
        df_am[f"breakeven_c_vs_{ref_policy}_q1"] = c_vs_ref
    return df_am


def subgroup_fairness_audit(
    df_test: pd.DataFrame,
    group_cols: list[str],
    policies_test: dict[str, np.ndarray],
    margins_test: np.ndarray,
) -> pd.DataFrame:
    """Audit referral rates, FPR, TPR, and precision across demographic subgroups."""
    y = np.asarray(margins_test, dtype=float) < 0
    rows = []
    for gcol in group_cols:
        if gcol not in df_test.columns:
            continue
        for gval, idx_pos in df_test.reset_index(drop=True).groupby(gcol, observed=True).indices.items():
            ix = np.asarray(idx_pos)
            yg = y[ix]
            for pnm, f in policies_test.items():
                fg = np.asarray(f, dtype=bool)[ix]
                n_g = len(ix)
                n_pos = int(yg.sum())
                n_neg = n_g - n_pos
                tp = int((fg & yg).sum())
                fp = int((fg & ~yg).sum())
                rows.append(
                    dict(
                        group_col=gcol,
                        group_val=str(gval),
                        policy=pnm,
                        n=n_g,
                        pct_neg_margin=round(yg.mean() * 100.0, 2),
                        pct_referred=round(fg.mean() * 100.0, 2),
                        fpr=round(fp / max(n_neg, 1) * 100.0, 2),
                        tpr=round(tp / max(n_pos, 1) * 100.0, 2),
                        precision=round(tp / max(int(fg.sum()), 1) * 100.0, 2),
                    )
                )
    return pd.DataFrame(rows)


def extract_tree_rules(clf: DecisionTreeClassifier, feature_names: list[str]) -> str:
    """Return human-readable text representation of a shallow decision tree."""
    return export_text(clf, feature_names=list(feature_names), decimals=2)


def compute_calibration_bins(probs_test: np.ndarray, margins_test: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute reliability diagram bins for calibration diagnostics."""
    y = (np.asarray(margins_test, dtype=float) < 0).astype(int)
    p = np.clip(np.asarray(probs_test, dtype=float), 0.0, 1.0)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
