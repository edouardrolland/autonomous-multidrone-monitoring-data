"""Run every analysis script and collect the results into outputs/.

Writes
    outputs/statistics.json    every reproduced number, machine-readable
    outputs/statistics.md      the same numbers as a readable summary
    outputs/figures/*.pdf      the seven data-driven figures of the manuscript

Usage: python3 reproduce_statistics.py
"""
import json
import sys

import reproduce_field_monitoring as monitoring
import reproduce_field_perception as perception
import reproduce_pso_validation as validation
import reproduce_scalability as scalability
from common import MISSIONS, OUT, REGIMES, SURFACES, ensure_outputs


def _k(d, key):
    """JSON round-trips integer keys as strings; accept either."""
    return d[key] if key in d else d[str(key)]


def summary_markdown(stats):
    L = ["# Reproduced statistics", "",
         "Every number below was recomputed from `data/` by the scripts in this directory.", ""]

    t = stats["pso_tuning_and_validation"]
    L += ["## PSO tuning (Table: specialist hyperparameter sets)", "",
          "| regime | best trial | w | c1 | c2 | c1/c2 | median Omega |",
          "|---|---|---|---|---|---|---|"]
    for r in REGIMES:
        b = t["tuning"][r]
        L.append(f"| {r} | {b['best_trial_id']} | {b['w']:.2f} | {b['c1']:.2f} | "
                 f"{b['c2']:.2f} | {b['c1_over_c2']:.1f} | {b['best_median_omega']:.3f} |")

    L += ["", "## PSO validation (paired on the 108 configuration means per regime)", "",
          "| regime | vs | n | win | median diff | Wilcoxon p | Holm p | TOST p | equivalent |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in REGIMES:
        for other, e in t["tests"][r]["comparisons"].items():
            hp = e.get("p_value_holm")
            L.append(f"| {r} | {other} | {e['n_pairs']} | "
                     f"{e['win_rate_matched_ge_other']:.2f} | {e['median_diff']:+.4f} | "
                     f"{e['p_value']:.3g} | {'-' if hp is None else f'{hp:.3g}'} | "
                     f"{e['tost']['p_tost']:.3g} | "
                     f"{'yes' if e['tost']['equivalent_alpha_0.05'] else 'no'} |")
    L += ["", "Tuned vs untuned default: " + ", ".join(
        f"{r} {t['tuned_vs_default_gain_pct'][r]:+.1f}%" for r in REGIMES), ""]

    s = stats["scalability"]
    cm = s["cost_model"]
    L += ["## Scalability", "",
          f"- {s['n_runs']} runs over {s['n_cells']} (herd, M) cells",
          f"- t_ref = {cm['t_ref_seconds']:.1f} s; normalised solve cost spans "
          f"{cm['normalised_cost_min']:.1f}-{cm['normalised_cost_max']:.1f}",
          f"- mean solve {s['wall_clock']['mean_solve_s']:.1f} s over the grid; largest cell "
          f"{s['wall_clock']['max_cell_mean_s']:.1f} s",
          f"- dispersion: normalised mean cost changes "
          f"{s['dispersion_effect_on_cost']['change_sigma10_to_sigma100_pct']:+.0f}% from "
          f"sigma = 10 m to 100 m", ""]

    p = stats["field_perception"]["per_mission"]
    L += ["## Field perception (Table: perception accuracy)", "",
          "| mission | N_GT | recall | precision | median position error | "
          "median body-axis error |", "|---|---|---|---|---|---|"]
    for m in MISSIONS + ["pooled"]:
        d = p[m]
        L.append(f"| {m} | {d['n_gt_instances']} | {d['recall_pct']:.0f}% | "
                 f"{d['precision_pct']:.0f}% | {d['median_position_error_m']:.2f} m | "
                 f"{d['median_axis_error_deg']:.1f} deg |")

    f = stats["field_monitoring"]
    cov, so = f["coverage"], f["standoff"]
    L += ["", "## Field coverage of the surfaces of interest", "",
          f"Pooled over the {cov['pooled']['n_frames']} monitoring frames of the four "
          "missions:", "",
          "| surface | V_k >= 0.5 | V_k >= 0.75 | V_k >= 0.9 |", "|---|---|---|---|"]
    for surf in SURFACES:
        L.append(f"| {surf.replace('_', ' ')} | "
                 + " | ".join(f"{cov['pooled'][f'pooled_time_share_V_{surf}_ge_{v:g}_pct']:.1f}%"
                              for v in (0.5, 0.75, 0.9)) + " |")
    L += ["", "| mission | median dorsal | median left flank | median right flank |",
          "|---|---|---|---|"]
    for m in MISSIONS:
        c = cov[m]
        L.append(f"| {m} | {c['median_V_dorsal']:.2f} | {c['median_V_left_flank']:.2f} | "
                 f"{c['median_V_right_flank']:.2f} |")

    L += ["", "## Stand-off compliance", "",
          f"- minimum stand-off respected at {so['compliant_pct']:.1f}% of "
          f"{so['n_instants']} evaluated drone-animal instances",
          f"- median deficit among breaches {so['median_deficit_m']:.1f} m", ""]
    return "\n".join(L)


def main():
    ensure_outputs()
    for mod in (validation, scalability, perception, monitoring):
        mod.main()
    stats = json.loads((OUT / "statistics.json").read_text())
    (OUT / "statistics.md").write_text(summary_markdown(stats))
    print(f"\nwrote {OUT / 'statistics.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
