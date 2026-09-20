"""Reproduce the PSO tuning and validation results (manuscript Section 4).

Reproduces
  Table  tuning-results    best (w, c1, c2) and median Omega per behaviour regime
  Table  cross-eval        config-mean Omega of every released controller on every regime
  Table  validation-tests  paired Wilcoxon + Holm + TOST(delta=0.005)
  Fig    tuning_convergence.pdf, tuning_coefficients.pdf, validation_boxplots.pdf

Usage: python3 reproduce_pso_validation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import (C_REGIME, DELTA_TOST, REGIMES, ensure_outputs, holm, load,
                    save_fig, tost, use_paper_style, plt)

CONTROLLERS = ["grazing", "travelling", "panic", "field", "generalist"]
NAME = {"grazing": "Grazing", "travelling": "Travelling", "panic": "Panic",
        "field": "Field", "generalist": "Default"}
# Box fill per controller, matching the manuscript figure.
C_BOX = {"grazing": "#4C9A52", "travelling": "#3B7DC4", "panic": "#C0504D",
         "field": "#E8913A", "generalist": "#A6A6A6"}


# --------------------------------------------------------------------------- tuning
def tuning_summary(trials, best):
    out = {}
    for regime in REGIMES:
        t = trials[trials.behaviour == regime]
        b = best[best.behaviour == regime].iloc[0]
        top = t.nlargest(10, "median_monitoring_quality")
        out[regime] = {
            "n_trials": int(len(t)),
            "best_trial_id": int(b.best_trial_id),
            "best_median_omega": float(b.median_monitoring_quality),
            "w": float(b.w), "c1": float(b.c1), "c2": float(b.c2),
            "c1_over_c2": float(b.c1 / b.c2),
            "top10_w_mean": float(top.w.mean()), "top10_w_sd": float(top.w.std()),
            "top10_c1_mean": float(top.c1.mean()), "top10_c1_sd": float(top.c1.std()),
            "top10_c2_mean": float(top.c2.mean()), "top10_c2_sd": float(top.c2.std()),
            "mean_trial_eval_minutes": float(t.eval_time_s.mean() / 60.0),
        }
    return out


def fig_tuning_convergence(trials, best):
    use_paper_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ymax = 0.0
    for regime in REGIMES:
        t = trials[trials.behaviour == regime].sort_values("trial_id")
        q = t.median_monitoring_quality.to_numpy()
        running = np.maximum.accumulate(q)
        ymax = max(ymax, running.max())
        ax.plot(t.trial_id, q, "o", ms=2.6, color=C_REGIME[regime], alpha=0.28, zorder=2)
        ax.plot(t.trial_id, running, "-", color=C_REGIME[regime], lw=1.8, zorder=3,
                label=NAME[regime])
        b = best[best.behaviour == regime].iloc[0]
        ax.plot(b.best_trial_id, b.median_monitoring_quality, "*", ms=12,
                color=C_REGIME[regime], markeredgecolor="black", markeredgewidth=0.5, zorder=4)
    ax.set_xlabel("TPE trial")
    ax.set_ylabel(r"Median monitoring quality $\widetilde{\Omega}$")
    ax.set_xlim(0, 101)
    ax.set_ylim(0.05, ymax * 1.08)
    ax.legend(loc="lower right", frameon=True)
    return save_fig(fig, "tuning_convergence.pdf")


def fig_tuning_coefficients(trials, best):
    use_paper_style()
    coeffs = [("w", r"Inertia weight $w$", (0, 1)),
              ("c1", r"Cognitive coefficient $c_1$", (0, 3)),
              ("c2", r"Social coefficient $c_2$", (0, 3))]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))
    fig.subplots_adjust(wspace=0.42, left=0.07, right=0.90, bottom=0.16, top=0.96)
    vmin = trials.median_monitoring_quality.min()
    vmax = trials.median_monitoring_quality.max()
    rng = np.random.default_rng(0)
    sc = None
    for ax, (col, label, ylim) in zip(axes, coeffs):
        for xpos, regime in enumerate(REGIMES):
            t = trials[trials.behaviour == regime]
            jitter = rng.uniform(-0.16, 0.16, len(t))
            sc = ax.scatter(np.full(len(t), xpos) + jitter, t[col],
                            c=t.median_monitoring_quality, cmap="viridis",
                            vmin=vmin, vmax=vmax, s=14, alpha=0.75,
                            edgecolors="none", zorder=2)
            bv = float(best[best.behaviour == regime].iloc[0][col])
            ax.plot(xpos, bv, "*", ms=15, color="white", markeredgecolor="black",
                    markeredgewidth=0.9, zorder=4)
            ax.annotate(f"{bv:.2f}", (xpos, bv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, fontweight="bold",
                        zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                            ec="none", alpha=0.85))
        ax.set_xticks(range(len(REGIMES)))
        ax.set_xticklabels([NAME[r] for r in REGIMES], rotation=12)
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, len(REGIMES) - 0.5)
        ax.grid(axis="x", visible=False)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.74])
    fig.colorbar(sc, cax=cax).set_label(r"$\widetilde{\Omega}$", rotation=90)
    return save_fig(fig, "tuning_coefficients.pdf")


# --------------------------------------------------------------------------- validation
def wide_config_means(cm):
    """{regime: DataFrame indexed by config_idx, one column per controller}."""
    out = {}
    for regime in REGIMES:
        d = cm[cm.behaviour == regime]
        out[regime] = (d.pivot(index="config_idx", columns="controller",
                               values="mean_monitoring_quality")
                        .reindex(columns=CONTROLLERS).sort_index())
    return out


def validation_tests(wide):
    """Paired Wilcoxon + TOST per (regime, other controller), with a pooled Holm correction.

    The Holm family is the nine comparisons of each matched specialist against the two
    mismatched specialists and against eta_field. The untuned default is excluded from it: it
    is a large-effect reference, not a member of the equivalence family.
    """
    tests, family = {}, []
    for regime in REGIMES:
        w = wide[regime]
        m = w[regime].to_numpy()
        tests[regime] = {"matched_controller": regime, "comparisons": {}}
        for other in CONTROLLERS:
            if other == regime:
                continue
            o = w[other].to_numpy()
            diffs = m - o
            p_tost, equiv = tost(diffs, DELTA_TOST)
            entry = {
                "n_pairs": int(len(diffs)),
                "win_rate_matched_ge_other": float(np.mean(m >= o)),
                "mean_diff": float(diffs.mean()),
                "median_diff": float(np.median(diffs)),
                "tost": {"delta": DELTA_TOST, "p_tost": p_tost,
                         "equivalent_alpha_0.05": equiv},
            }
            if np.all(diffs == 0):
                entry.update({"statistic": 0.0, "p_value": 1.0,
                              "significant_alpha_0.05": False})
            else:
                stat, p = wilcoxon(m, o, zero_method="wilcox", alternative="two-sided")
                entry.update({"statistic": float(stat), "p_value": float(p),
                              "significant_alpha_0.05": bool(p < 0.05)})
            tests[regime]["comparisons"][other] = entry
            if other != "generalist":
                family.append((regime, other, entry["p_value"]))

    adj = holm([p for _, _, p in family])
    for (regime, other, _), p_adj in zip(family, adj):
        e = tests[regime]["comparisons"][other]
        e["p_value_holm"] = float(p_adj)
        e["significant_holm_0.05"] = bool(p_adj < 0.05)
    return tests


def cross_evaluation(wide):
    """Cross-evaluation table: the manuscript reports the median over the 108 scenarios."""
    return {"median": {regime: {c: float(wide[regime][c].median()) for c in CONTROLLERS}
                       for regime in REGIMES},
            "mean": {regime: {c: float(wide[regime][c].mean()) for c in CONTROLLERS}
                     for regime in REGIMES}}


def fig_validation_boxplots(wide):
    """Scenario-level Omega per controller, one panel per behavioural state.

    Boxes only: at n = 108 per box the individual scenarios overplot into a solid band and hide
    the quartiles, which are what the figure is read for. The dashed rule marks the matched
    specialist's median, so the eye can carry it across the other controllers.
    """
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        w = wide[regime]
        bp = ax.boxplot([w[c].to_numpy() for c in CONTROLLERS], widths=0.62,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.2),
                        whiskerprops=dict(color="0.25", lw=0.9),
                        capprops=dict(color="0.25", lw=0.9),
                        boxprops=dict(lw=0.9))
        for patch, c in zip(bp["boxes"], CONTROLLERS):
            patch.set_facecolor(C_BOX[c])
            patch.set_edgecolor("0.2")
        ax.axhline(float(w[regime].median()), color="#2e7d32", ls="--", lw=0.9, zorder=0)
        ax.set_xticks(range(1, len(CONTROLLERS) + 1))
        ax.set_xticklabels([NAME[c] for c in CONTROLLERS], rotation=30, ha="right")
        ax.set_title(NAME[regime])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(r"Validation monitoring quality $\Omega$")
    fig.tight_layout()
    return save_fig(fig, "validation_boxplots_pretuning.pdf")


# --------------------------------------------------------------------------- driver
def figures():
    trials = load("pso_tuning_trials.csv")
    best = load("pso_tuning_best.csv")
    wide = wide_config_means(load("pso_validation_config_means.csv"))
    return [fig_tuning_convergence(trials, best),
            fig_tuning_coefficients(trials, best),
            fig_validation_boxplots(wide)]


def statistics():
    trials = load("pso_tuning_trials.csv")
    best = load("pso_tuning_best.csv")
    runs = load("pso_validation_runs.csv")
    cm = load("pso_validation_config_means.csv")
    wide = wide_config_means(cm)

    # the config mean is the statistical unit: check the published means are what the
    # published runs average to, so the two levels cannot silently disagree
    recomputed = (runs.groupby(["behaviour", "controller", "config_idx"])
                      .monitoring_quality.mean().rename("mean_monitoring_quality")
                      .reset_index())
    merged = cm.merge(recomputed, on=["behaviour", "controller", "config_idx"],
                      suffixes=("_published", "_recomputed"))
    max_dev = float((merged.mean_monitoring_quality_published
                     - merged.mean_monitoring_quality_recomputed).abs().max())

    res = {
        "n_validation_runs": int(len(runs)),
        "n_config_means": int(len(cm)),
        "config_mean_max_abs_deviation_runs_vs_published": max_dev,
        "tuning": tuning_summary(trials, best),
        "cross_evaluation_omega": cross_evaluation(wide),
        "tests": validation_tests(wide),
    }
    res["tuned_vs_default_gain_pct"] = {
        regime: 100.0 * (wide[regime][regime].mean() / wide[regime]["generalist"].mean() - 1.0)
        for regime in REGIMES
    }
    return res


def main():
    ensure_outputs()
    print("PSO tuning and validation")
    res = statistics()
    for regime, t in res["tuning"].items():
        print(f"  {regime:<11s} best trial #{t['best_trial_id']:3d}  "
              f"(w,c1,c2)=({t['w']:.3f},{t['c1']:.3f},{t['c2']:.3f})  "
              f"Omega_med={t['best_median_omega']:.4f}")
    print(f"  config-mean check: max |published - recomputed| = "
          f"{res['config_mean_max_abs_deviation_runs_vs_published']:.2e}")
    for regime in REGIMES:
        gain = res["tuned_vs_default_gain_pct"][regime]
        print(f"  {regime:<11s} tuned vs default: {gain:+.1f}%")
        for other, e in res["tests"][regime]["comparisons"].items():
            holm_p = e.get("p_value_holm")
            print(f"    vs {other:<11s} n={e['n_pairs']:3d} "
                  f"win={e['win_rate_matched_ge_other']:.3f} "
                  f"p={e['p_value']:.3g} "
                  f"p_holm={'n/a' if holm_p is None else f'{holm_p:.3g}'} "
                  f"TOST p={e['tost']['p_tost']:.3g} "
                  f"equivalent={e['tost']['equivalent_alpha_0.05']}")
    from common import write_results
    write_results("pso_tuning_and_validation", res)
    figures()


if __name__ == "__main__":
    main()
