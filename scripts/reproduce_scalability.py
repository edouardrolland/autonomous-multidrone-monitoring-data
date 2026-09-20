"""Reproduce the scalability results (manuscript Section 6).

Reproduces
  Fig  sigma_axis_surfaces.pdf  per-surface coverage envelope over (sigma, M) for each N_a
  Fig  solve_time.pdf           normalised solve cost against M and against N_a
  Text solve cost against M and N_a, and the effect of spatial spread on it
       (Table: effect of spatial spread on computational cost)

Usage: python3 reproduce_scalability.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (SIM, ensure_outputs, load, plt, save_fig, use_paper_style, write_results)

HERD_UNIT = ["behaviour", "N_a", "sigma", "herd_idx", "M"]
VALUE_COLS = ["V1", "V2", "V3", "minV", "per_animal_min_mean", "per_animal_min_p10",
              "Omega", "Delta", "Lambda_dist", "Lambda_idle", "t_solve"]
FIELD_SIGMA_MAX = 30.0          # dispersion range representative of the field site
FIELD_NA_MIN = 25               # "a few dozen to a hundred animals"


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: mean over PSO repetitions -> one row per (cell, herd, M)."""
    agg = runs.groupby(HERD_UNIT, dropna=False)[VALUE_COLS].mean().reset_index()
    agg["n_reps"] = runs.groupby(HERD_UNIT, dropna=False).size().values
    return agg


# --------------------------------------------------------------------------- figures
def fig_sigma_axis_surfaces(agg):
    """Rows = surface class, columns = herd size; each cell is a (sigma x M) heatmap."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "mathtext.fontset": "dejavuserif", "font.size": 8,
                         "axes.linewidth": 0.6, "axes.grid": False,
                         "figure.constrained_layout.use": True})
    agg = agg.copy()
    # The two flanks are the same measurement mirrored; one lateral row states that once.
    agg["Vflank"] = agg[["V2", "V3"]].mean(axis=1)
    surfaces = [("V1", "$V_1$ dorsal\nspread $\\sigma$ (m)"),
                ("Vflank", "$V_{2,3}$ lateral\nspread $\\sigma$ (m)")]
    herd_sizes = sorted(agg.N_a.unique())

    fig, axes = plt.subplots(len(surfaces), len(herd_sizes),
                             figsize=(9.9, 2.055 * len(surfaces) + 1.1),
                             sharex=True, sharey=True, squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.015, hspace=0.03)
    mesh = None
    for r, (col, slabel) in enumerate(surfaces):
        for c, N_a in enumerate(herd_sizes):
            ax = axes[r][c]
            grid = (agg[agg.N_a == N_a]
                    .pivot_table(index="sigma", columns="M", values=col, aggfunc="median"))
            values = grid.to_numpy()
            mesh = ax.imshow(values, origin="upper", aspect="auto", cmap="viridis",
                             vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_aspect("equal")
            ax.set_xticks(range(grid.shape[1]))
            ax.set_xticklabels([int(x) for x in grid.columns])
            ax.set_yticks(range(grid.shape[0]))
            ax.set_yticklabels([f"{y:g}" for y in grid.index])
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", lw=0.35)
            ax.tick_params(which="minor", length=0)
            ax.tick_params(labelsize=8, pad=1.0)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    v = values[i, j]
                    if np.isnan(v):
                        continue
                    txt = "1" if v >= 0.995 else f"{v:.2f}".lstrip("0")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.0,
                            color="white" if v < 0.55 else "black")
            if r == 0:
                ax.set_title(rf"$N_a = {int(N_a)}$", fontsize=11)
            if r == len(surfaces) - 1:
                ax.set_xlabel("$M$", fontsize=11)
            if c == 0:
                ax.set_ylabel(slabel, fontsize=10.5)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal",
                        fraction=0.022, pad=0.02, aspect=60)
    cbar.set_label(r"Herd-median $V_k$", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)
    path = save_fig(fig, "sigma_axis_surfaces.pdf")
    plt.rcParams.update({"figure.constrained_layout.use": False})
    return path


def fig_solve_time(runs, t_ref_seconds):
    use_paper_style()
    t_ref = runs[(runs.M == 1) & (runs.N_a == runs.N_a.min())].t_solve.mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

    sizes = sorted(runs.N_a.unique())
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(sizes)))
    for N_a, colour in zip(sizes, colours):
        sub = runs[runs.N_a == N_a].groupby("M").t_solve
        mean, lo, hi = sub.mean() / t_ref, sub.quantile(0.10) / t_ref, sub.quantile(0.90) / t_ref
        ax1.fill_between(mean.index, lo, hi, color=colour, alpha=0.15, lw=0)
        ax1.plot(mean.index, mean, "o-", color=colour, ms=3, lw=1.0, label=rf"${N_a}$")
    ax1.set_xlabel("Monitoring drones $M$")
    ax1.set_ylabel(r"Normalised solve cost $t / t_{\mathrm{ref}}$")
    ax1.set_xticks(sorted(runs.M.unique()))
    ax1.legend(title=r"Herd size $N_a$", ncol=2, frameon=False, fontsize="small",
               title_fontsize="small")

    swarms = sorted(runs.M.unique())[::2]
    colours = plt.cm.plasma(np.linspace(0.05, 0.75, len(swarms)))
    for M, colour in zip(swarms, colours):
        sub = runs[runs.M == M].groupby("N_a").t_solve
        mean, lo, hi = sub.mean() / t_ref, sub.quantile(0.10) / t_ref, sub.quantile(0.90) / t_ref
        ax2.fill_between(mean.index, lo, hi, color=colour, alpha=0.15, lw=0)
        ax2.plot(mean.index, mean, "s-", color=colour, ms=3, lw=1.0, label=rf"${M}$")
    ax2.set_xlabel("Herd size $N_a$")
    ax2.set_xticks(sizes)
    ax2.legend(title=r"Drones $M$", ncol=2, frameon=False, fontsize="small",
               title_fontsize="small")

    for ax in (ax1, ax2):
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelleft=True)
        sec = ax.secondary_yaxis("right", functions=(lambda y: y * t_ref_seconds,
                                                     lambda y: y / t_ref_seconds))
        sec.set_ylabel("Wall-clock time on i7-13700HX (s)", fontsize="small")
    return save_fig(fig, "solve_time.pdf")


# --------------------------------------------------------------------------- statistics
def cost_model(runs):
    """Least squares t/t_ref ~ 1 + M + N_a + M*N_a over every individual run."""
    t_ref = runs[(runs.M == 1) & (runs.N_a == runs.N_a.min())].t_solve.mean()
    y = (runs.t_solve / t_ref).to_numpy()
    X = np.column_stack([np.ones(len(runs)), runs.M, runs.N_a, runs.M * runs.N_a])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var()
    cell = runs.groupby(["M", "N_a"]).t_solve.mean() / t_ref
    return {"t_ref_seconds": float(t_ref),
            "intercept": float(beta[0]), "beta_M": float(beta[1]),
            "beta_Na": float(beta[2]), "beta_MNa": float(beta[3]),
            "r_squared": float(r2),
            "normalised_cost_min": float(cell.min()),
            "normalised_cost_max": float(cell.max())}


def field_operating_point(agg):
    sub = agg[(agg.sigma <= FIELD_SIGMA_MAX) & (agg.N_a >= FIELD_NA_MIN)]
    med = sub.groupby("M").minV.median()
    return {"sigma_max_m": FIELD_SIGMA_MAX, "N_a_min": FIELD_NA_MIN,
            "median_minV_M4": float(med.get(4, np.nan)),
            "median_minV_M6": float(med.get(6, np.nan)),
            "median_minV_M9": float(med.get(9, np.nan)),
            "gain_M6_to_M9": float(med.get(9, np.nan) - med.get(6, np.nan))}


def dispersion_effect(runs):
    t_ref = runs[(runs.M == 1) & (runs.N_a == runs.N_a.min())].t_solve.mean()
    m = runs.groupby("sigma").t_solve.mean() / t_ref
    return {"normalised_mean_cost_by_sigma": {float(k): float(v) for k, v in m.items()},
            "change_sigma10_to_sigma100_pct":
                float(100.0 * (m.iloc[-1] / m.iloc[0] - 1.0))}


def statistics():
    runs = load("scalability_runs.csv")
    agg = aggregate(runs)
    manifest = json.loads((SIM / "manifest_standard.json").read_text())
    res = {
        "n_runs": int(len(runs)),
        "n_cells": int(len(agg)),
        "factors": {"N_a": sorted(int(x) for x in runs.N_a.unique()),
                    "M": sorted(int(x) for x in runs.M.unique()),
                    "sigma": sorted(float(x) for x in runs.sigma.unique()),
                    "herd_seeds": int(runs.herd_idx.nunique()),
                    "reps_per_herd": int(runs.rep.nunique())},
        "converged_fraction": float((runs.status == runs.status.mode()[0]).mean()),
        "cost_model": cost_model(runs),
        "field_operating_point": field_operating_point(agg),
        "dispersion_effect_on_cost": dispersion_effect(runs),
        "wall_clock": {"mean_solve_s": float(runs.t_solve.mean()),
                       "p5_solve_s": float(runs.t_solve.quantile(0.05)),
                       "p95_solve_s": float(runs.t_solve.quantile(0.95)),
                       "max_cell_mean_s": float(runs.groupby(["M", "N_a"])
                                                .t_solve.mean().max()),
                       "total_cpu_hours": float(runs.t_solve.sum() / 3600.0)},
        "manifest": manifest,
    }
    return res, runs, agg


def figures():
    runs = load("scalability_runs.csv")
    agg = aggregate(runs)
    t_ref = runs[(runs.M == 1) & (runs.N_a == runs.N_a.min())].t_solve.mean()
    return [fig_sigma_axis_surfaces(agg), fig_solve_time(runs, t_ref)]


def main():
    ensure_outputs()
    print("Scalability")
    res, runs, agg = statistics()
    cm = res["cost_model"]
    print(f"  runs={res['n_runs']}  cells={res['n_cells']}")
    print(f"  t_ref = {cm['t_ref_seconds']:.2f} s  (mean of the M=1, N_a={min(res['factors']['N_a'])} cell)")
    print(f"  t/t_ref = {cm['intercept']:.3f} + {cm['beta_M']:.3f} M "
          f"+ {cm['beta_Na']:.4f} N_a + {cm['beta_MNa']:.4f} M N_a   R2={cm['r_squared']:.2f}")
    print(f"  normalised cost range: {cm['normalised_cost_min']:.1f} - "
          f"{cm['normalised_cost_max']:.1f} t_ref")
    fop = res["field_operating_point"]
    print(f"  field regime (sigma<={fop['sigma_max_m']:.0f} m): median min_k V_k "
          f"M=4 {fop['median_minV_M4']:.3f}  M=6 {fop['median_minV_M6']:.3f}  "
          f"M=9 {fop['median_minV_M9']:.3f}  (+{fop['gain_M6_to_M9']:.3f} for the last three)")
    print(f"  dispersion: normalised mean cost changes "
          f"{res['dispersion_effect_on_cost']['change_sigma10_to_sigma100_pct']:+.0f}% "
          f"from sigma=10 to sigma=100 m")
    wc = res["wall_clock"]
    print(f"  wall clock: mean {wc['mean_solve_s']:.1f} s "
          f"(p5-p95 {wc['p5_solve_s']:.1f}-{wc['p95_solve_s']:.1f} s), "
          f"largest cell {wc['max_cell_mean_s']:.1f} s, {wc['total_cpu_hours']:.0f} CPU-hours")
    write_results("scalability", res)
    figures()


if __name__ == "__main__":
    main()
