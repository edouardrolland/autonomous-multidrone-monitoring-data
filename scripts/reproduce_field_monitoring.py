"""Reproduce the field coverage and stand-off results (manuscript Section 6).

Reproduces
  Fig   coverage_distribution.pdf  share of monitoring time with V_k >= v, per mission
  Fig   coverage_timeline_vk.pdf   mean flank coverage over each mission: the reference the
                                   PSO reaches on the true herd, the configuration it
                                   commanded on the perceived herd, and what was flown

Usage: python3 reproduce_field_monitoring.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (C_SURFACE, MISSIONS, SURFACES, TAU, ensure_outputs, load, plt, save_fig,
                    use_paper_style, write_results)

GRID = np.linspace(0, 1, 201)
SURF_LABEL = {"dorsal": "dorsal", "left_flank": "left flank", "right_flank": "right flank"}


# --------------------------------------------------------------------------- coverage
def frame_coverage(cov, mid, on_target_only=False):
    """(F, 3) herd-average V_k per frame: V_k = mean over animals of max over drones.
    """
    d = cov[cov.mission_id == mid]
    if on_target_only:
        d = d[d.on_target_herd]
    per_animal = (d.pivot_table(index=["gt_frame", "animal_idx"], columns="surface",
                                values="visible_fraction", aggfunc="max")
                   .reindex(columns=SURFACES))
    return per_animal.groupby(level="gt_frame").mean().to_numpy()


def animal_coverage_cube(cov, mid):
    """(A, D, 3) visible fraction per animal-instant, drone and surface."""
    d = cov[cov.mission_id == mid]
    cube = (d.pivot_table(index=["gt_frame", "animal_idx"], columns=["drone_idx", "surface"],
                          values="visible_fraction")
             .sort_index(axis=1))
    drones = sorted(d.drone_idx.unique())
    return np.stack([cube[[(j, s) for s in SURFACES]].to_numpy() for j in drones], axis=1)


def coverage_stats(cov):
    out = {}
    for mid in MISSIONS:
        V = frame_coverage(cov, mid, on_target_only=True)
        cube = animal_coverage_cube(cov, mid)
        best = np.nanmax(cube, axis=1)                     # (A, 3) best drone per surface
        out[mid] = {
            "n_frames": int(V.shape[0]),
            "minutes": float(V.shape[0] * 8 / 60),
            "n_animal_instants": int(best.shape[0]),
            "n_monitoring_drones": int(cube.shape[1]),
        }
        for i, s in enumerate(SURFACES):
            out[mid][f"median_V_{s}"] = float(np.median(V[:, i]))
            out[mid][f"mean_V_{s}"] = float(np.mean(V[:, i]))
            out[mid][f"time_share_V_{s}_ge_tau_pct"] = float(100.0 * (V[:, i] >= TAU).mean())
        out[mid]["all_three_surfaces_ge_tau_pct"] = float(
            100.0 * ((best >= TAU).sum(axis=1) == 3).mean())
    pooled = {}
    for i, s in enumerate(SURFACES):
        vals = [out[m][f"time_share_V_{s}_ge_tau_pct"] for m in MISSIONS]
        pooled[f"time_share_V_{s}_ge_tau_pct_range"] = [float(min(vals)), float(max(vals))]
        med = [out[m][f"median_V_{s}"] for m in MISSIONS]
        pooled[f"median_V_{s}_range"] = [float(min(med)), float(max(med))]
    allV = np.vstack([frame_coverage(cov, m) for m in MISSIONS])
    pooled["n_frames"] = int(allV.shape[0])
    for i, s in enumerate(SURFACES):
        for v in (0.5, 0.75, 0.9):
            pooled[f"pooled_time_share_V_{s}_ge_{v:g}_pct"] = float(
                100.0 * (allV[:, i] >= v).mean())
    out["pooled"] = pooled
    return out


# --------------------------------------------------------------------------- stand-off
def standoff_stats():
    s = load("standoff_instants.csv")
    breaches = s[~s.compliant]
    per = {mid: {"n_instants": int((s.mission_id == mid).sum()),
                 "compliant_pct": float(100.0 * s[s.mission_id == mid].compliant.mean()),
                 "median_closest_m": float(
                     s[s.mission_id == mid].closest_animal_distance_m.median()),
                 "min_closest_m": float(
                     s[s.mission_id == mid].closest_animal_distance_m.min())}
           for mid in MISSIONS}
    return {"n_instants": int(len(s)),
            "compliant_pct": float(100.0 * s.compliant.mean()),
            "n_breaches": int(len(breaches)),
            "median_deficit_m": float(breaches.deficit_m.median()),
            "p90_deficit_m": float(breaches.deficit_m.quantile(0.90)),
            "max_deficit_m": float(breaches.deficit_m.max()),
            "median_closest_m": float(s.closest_animal_distance_m.median()),
            "p5_closest_m": float(s.closest_animal_distance_m.quantile(0.05)),
            "per_mission": per}


# --------------------------------------------------------------------------- figures
def fig_coverage_distribution(cov):
    use_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    n_mon = {m: int(load("missions.csv").set_index("mission_id")
                    .loc[m, "n_monitoring_drones"]) for m in MISSIONS}
    for ax, mid in zip(axes, MISSIONS):
        V = frame_coverage(cov, mid, on_target_only=True)
        for i, s in enumerate(SURFACES):
            surv = (V[:, i][:, None] >= GRID[None, :]).mean(axis=0)
            ax.plot(100 * GRID, 100 * surv, color=C_SURFACE[s], lw=1.8, zorder=3,
                    label=SURF_LABEL[s])
            ax.plot([100 * TAU], [100 * float((V[:, i] >= TAU).mean())], marker="o",
                    ms=4.5, color=C_SURFACE[s], zorder=5, mec="white", mew=0.9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 103)
        ax.set_xticks([0, 25, 50, 75, 100])     # as published; matplotlib would pick 0-20-...-100
        ax.set_title(f"{mid}  ($M = {n_mon[mid]}$)", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Monitoring time with\n$V_k \\geq v$ (%)", fontsize=9)
    fig.supxlabel("Coverage level $v$ (%)", fontsize=9, y=-0.04)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    return save_fig(fig, "coverage_distribution.pdf")


def fig_coverage_timeline_vk():
    """Mean flank coverage over each mission: reference, commanded, and realised.

    reference  V_{2,3} at the pose the PSO reaches when given the TRUE herd, solved once per
               real field solve (mean of five seeded replicates, +-1 SD band). It is the Omega
               optimum, not an upper bound on V_{2,3}: Omega also pays for stand-off, framing
               and the dorsal surface, so the realised flanks can and do exceed it.
    commanded  V_{2,3} of the configuration the controller selected, scored on the herd it
               perceived. A different measurement plane, so it is drawn as markers.
    realised   V_{2,3} at the poses the drones actually occupied, on the true herd.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    C_CEIL, C_R, C_T = "#009E73", "#0072B2", "#D55E00"      # Okabe-Ito
    C_TRANSIT, C_BREAK = "0.88", "#DCD2EA"
    A_TRANSIT, A_BREAK = 0.5, 0.36
    BAND_FS, BAND_C = 7.8, "0.45"
    YMAX, YPAD = 1.0, 0.035
    # Editorial annotations, not measurements: what the operators recorded as the reason a
    # stretch carries no solves. The spans themselves are derived from the solve cadence.
    GAP_TEXT = {"F3": "scout switched\nto a new herd",
                "F2": "ground station froze\n(overheating)"}
    EXTRA_TRANSIT = {"F3": [(246.0, 310.0, "transit to\nthe new herd")]}

    use_paper_style()
    plt.rcParams.update({"axes.grid": True, "grid.color": "0.93", "grid.linewidth": 0.5,
                         "axes.axisbelow": True})
    reps = load("monitoring_timeline_reps.csv")
    cmd = load("coverage_commanded.csv")
    missions = load("missions.csv").set_index("mission_id")

    def lat(df, prefix):
        return (df[f"{prefix}_left_flank"] + df[f"{prefix}_right_flank"]) / 2.0

    reps = reps.assign(lat_ceiling=lat(reps, "v_ceiling"), lat_realised=lat(reps, "v_realised"))
    curve = (reps.groupby(["mission_id", "solve"])
                 .agg(t_rel_s=("t_rel_s", "first"), ceiling=("lat_ceiling", "mean"),
                      sd=("lat_ceiling", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
                      realised=("lat_realised", "first")).reset_index())
    cmd = cmd.assign(lat_commanded=lat(cmd, "v_commanded"))

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.945, bottom=0.185,
                        wspace=0.075, hspace=0.36)

    for ax, mid in zip(axes.ravel(), MISSIONS):
        d = curve[curve.mission_id == mid].sort_values("t_rel_s")
        c = cmd[cmd.mission_id == mid].sort_values("t_rel_s")
        t0 = float(d.t_rel_s.iloc[0])
        t = d.t_rel_s.to_numpy() - t0
        on0 = float(missions.loc[mid, "onstation_start_s"]) - t0
        on1 = float(missions.loc[mid, "onstation_end_s"]) - t0

        # Break the line only across gaps abnormal FOR THIS MISSION: the solve cadence itself
        # runs from 6 to 19 s with herd size, so a fixed threshold flags F2's normal rhythm.
        gap = max(25.0, 2.5 * float(np.median(np.diff(t)))) if len(t) > 1 else np.inf
        gaps = [(t[k - 1], t[k]) for k in range(1, len(t)) if t[k] - t[k - 1] > gap]
        brk = [k for k in range(1, len(t)) if t[k] - t[k - 1] > gap]
        tt = np.insert(t, brk, [(g0 + g1) / 2 for g0, g1 in gaps])
        ceil = np.insert(d.ceiling.to_numpy(), brk, np.nan)
        sd = np.insert(d.sd.to_numpy(), brk, np.nan)
        real = np.insert(d.realised.to_numpy(), brk, np.nan)

        if t.min() < on0:
            ax.axvspan(t.min(), on0, color=C_TRANSIT, alpha=A_TRANSIT, lw=0, zorder=0)
        if t.max() > on1:
            ax.axvspan(on1, t.max(), color=C_TRANSIT, alpha=A_TRANSIT, lw=0, zorder=0)
        for x0, x1, lab in EXTRA_TRANSIT.get(mid, []):
            ax.axvspan(x0, x1, color=C_TRANSIT, alpha=A_TRANSIT, lw=0, zorder=0)
            ax.text(x0 + 0.35 * (x1 - x0), 0.42 * YMAX, lab, ha="center", va="center",
                    rotation=90, fontsize=BAND_FS, color=BAND_C, style="italic", zorder=6)
        for g0, g1 in gaps:
            ax.axvspan(g0, g1, color=C_BREAK, alpha=A_BREAK, lw=0, zorder=0)
            if g1 - g0 > 40:
                ax.text((g0 + g1) / 2, YMAX / 2, GAP_TEXT.get(mid, "no solves\nlogged"),
                        ha="center", va="center", rotation=90, fontsize=BAND_FS,
                        color=BAND_C, style="italic", zorder=6)

        ax.fill_between(tt, ceil - sd, ceil + sd, color=C_CEIL, alpha=0.20, lw=0, zorder=2)
        ax.plot(tt, ceil, color=C_CEIL, lw=1.5, zorder=3)
        ax.plot(tt, real, color=C_R, lw=2.1, zorder=4)
        ax.plot(c.t_rel_s - t0, c.lat_commanded, ls="none", marker="v", ms=2.6, color=C_T,
                zorder=5)
        ax.set_title(f"{mid}  ($M = {int(missions.loc[mid, 'n_monitoring_drones'])}$)",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(0, YMAX + YPAD)
        ax.set_xlim(t.min(), t.max())
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(labelsize=9)

    fig.supxlabel("Time since first solve (s)", fontsize=10, y=0.085)
    fig.supylabel("$V_{2,3}$ — mean of both flanks", fontsize=10, x=0.018)
    handles = [
        plt.Line2D([], [], color=C_CEIL, lw=1.5,
                   label="$\\Omega$-optimal reference (mean $\\pm$ SD)"),
        plt.Line2D([], [], color=C_R, lw=2.1, label="Realised"),
        plt.Line2D([], [], color=C_T, ls="none", marker="v", ms=2.6, label="Commanded"),
        Patch(facecolor=C_TRANSIT, alpha=A_TRANSIT, label="Transit"),
        Patch(facecolor=C_BREAK, alpha=A_BREAK, label="Solving interruption")]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.005),
               frameon=False, fontsize=8.5)
    return save_fig(fig, "coverage_timeline_vk.pdf")


def figures():
    cov = load("surface_coverage.csv")
    return [fig_coverage_distribution(cov), fig_coverage_timeline_vk()]


def statistics():
    return {"coverage": coverage_stats(load("surface_coverage.csv")),
            "standoff": standoff_stats()}


def main():
    ensure_outputs()
    print("Field controller, coverage and stand-off")
    res = statistics()

    for mid in MISSIONS:
        c = res["coverage"][mid]
        print(f"  {mid} coverage: median dorsal {c['median_V_dorsal']:.2f}  "
              f"left {c['median_V_left_flank']:.2f}  right {c['median_V_right_flank']:.2f}  "
              f"all-three at tau {c['all_three_surfaces_ge_tau_pct']:.0f}%")

    so = res["standoff"]
    print(f"  stand-off: {so['compliant_pct']:.1f}% of {so['n_instants']} drone-instants "
          f"compliant; median deficit {so['median_deficit_m']:.2f} m")
    write_results("field_monitoring", res)
    figures()


if __name__ == "__main__":
    main()
