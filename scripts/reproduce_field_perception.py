"""Reproduce the field perception and localisation results (manuscript Section 7.1).

Reproduces
  Table field-localisation  N_GT, recall, precision, median position error, median body-axis
                            error per mission

Usage: python3 reproduce_field_perception.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import (MATCH_GATE_M, MISSIONS, axis_error_deg, ensure_outputs,
                    haversine_m, load, write_results)

SYNC_MU_MIN = 1.5        # m, minimum common shift worth flagging
SYNC_COH_MIN = 0.6       # |mean| / (|mean| + RMS residual): ->1 rigid, ->0 random
SYNC_MIN_PAIRS = 3
C_MISSION = {"F1": "#0072B2", "F2": "#CC79A7", "F3": "#D55E00", "F4": "#009E73"}


def match_frame(gt, pred, gate_m=MATCH_GATE_M):
    """Minimum-cost association within `gate_m`. Returns a list of matched row pairs.
    """
    if len(gt) == 0 or len(pred) == 0:
        return []
    D = haversine_m(gt.lat.to_numpy()[:, None], gt.lon.to_numpy()[:, None],
                    pred.lat.to_numpy()[None, :], pred.lon.to_numpy()[None, :])
    C = np.where(D <= gate_m, D, gate_m + 1e6)
    ri, ci = linear_sum_assignment(C)
    return [(int(r), int(c), float(D[r, c])) for r, c in zip(ri, ci) if D[r, c] <= gate_m]


def is_sync_frame(gt, pred, pairs):

    if len(pairs) < SYNC_MIN_PAIRS:
        return False
    v = []
    for r, c, _ in pairs:
        glat, glon = gt.lat.iloc[r], gt.lon.iloc[r]
        plat, plon = pred.lat.iloc[c], pred.lon.iloc[c]
        v.append(((glat - plat) * 111000.0,
                  (glon - plon) * 111000.0 * np.cos(np.radians(glat))))
    v = np.array(v)
    mu = v.mean(axis=0)
    mu_mag = float(np.hypot(*mu))
    resid = float(np.sqrt(np.mean(np.sum((v - mu) ** 2, axis=1))))
    coh = mu_mag / (mu_mag + resid + 1e-9)
    return mu_mag >= SYNC_MU_MIN and coh >= SYNC_COH_MIN


def build_matches():
    """One row per matched (label, prediction) pair, with its position and axis error."""
    labels = load("perception_gt_labels.csv")
    preds = load("perception_predictions.csv")
    rows = []
    frame_rows = []
    for mid in MISSIONS:
        L = labels[labels.mission_id == mid]
        P = preds[preds.mission_id == mid]
        for frame in sorted(set(L.frame) | set(P.frame)):
            gt = L[L.frame == frame].reset_index(drop=True)
            pr = P[P.frame == frame].reset_index(drop=True)
            pairs = match_frame(gt, pr)
            sync = is_sync_frame(gt, pr, pairs)
            frame_rows.append({"mission_id": mid, "frame": frame, "n_gt": len(gt),
                               "n_pred": len(pr), "n_matched": len(pairs),
                               "sync_offset_frame": sync})
            for r, c, d in pairs:
                a_gt = gt.axis_bearing_deg.iloc[r]
                a_pr = pr.axis_bearing_deg.iloc[c]
                rows.append({
                    "mission_id": mid, "frame": frame,
                    "label_idx": int(gt.label_idx.iloc[r]),
                    "prediction_idx": int(pr.prediction_idx.iloc[c]),
                    "position_error_m": d,
                    "axis_error_deg": (axis_error_deg(a_gt, a_pr)
                                       if np.isfinite(a_gt) and np.isfinite(a_pr) else np.nan),
                    "sync_offset_frame": sync,
                })
    return pd.DataFrame(rows), pd.DataFrame(frame_rows)


def mission_table(matches, frames):
    out = {}
    for mid in MISSIONS:
        f = frames[frames.mission_id == mid]
        m = matches[matches.mission_id == mid]
        clean = m[~m.sync_offset_frame]
        out[mid] = {
            "n_frames": int(len(f)),
            "n_gt_instances": int(f.n_gt.sum()),
            "n_predictions": int(f.n_pred.sum()),
            "n_matched": int(f.n_matched.sum()),
            "recall_pct": float(100.0 * f.n_matched.sum() / f.n_gt.sum()),
            "precision_pct": float(100.0 * f.n_matched.sum() / f.n_pred.sum()),
            "n_sync_offset_frames": int(f.sync_offset_frame.sum()),
            "median_position_error_m": float(clean.position_error_m.median()),
            "mean_position_error_m": float(clean.position_error_m.mean()),
            "p90_position_error_m": float(clean.position_error_m.quantile(0.90)),
            "median_axis_error_deg": float(m.axis_error_deg.median()),
            "n_axis_pairs": int(m.axis_error_deg.notna().sum()),
        }
    clean_all = matches[~matches.sync_offset_frame]
    out["pooled"] = {
        "n_frames": int(len(frames)),
        "n_gt_instances": int(frames.n_gt.sum()),
        "recall_pct": float(100.0 * frames.n_matched.sum() / frames.n_gt.sum()),
        "precision_pct": float(100.0 * frames.n_matched.sum() / frames.n_pred.sum()),
        "median_position_error_m": float(clean_all.position_error_m.median()),
        "median_axis_error_deg": float(matches.axis_error_deg.median()),
        "axis_error_under_15deg_pct": float(100.0 * (matches.axis_error_deg < 15).mean()),
        "axis_error_under_30deg_pct": float(100.0 * (matches.axis_error_deg < 30).mean()),
    }
    return out

def figures():
    """The manuscript reports perception as a table, not a figure."""
    return []


def statistics():
    matches, frames = build_matches()
    ensure_outputs()
    from common import OUT
    matches.to_csv(OUT / "perception_matches.csv", index=False)
    return {"per_mission": mission_table(matches, frames)}


def main():
    ensure_outputs()
    print("Field perception")
    res = statistics()
    print(f"  {'mission':<8}{'N_GT':>6}{'recall':>9}{'precision':>11}"
          f"{'pos err':>10}{'axis err':>10}")
    for mid in MISSIONS + ["pooled"]:
        t = res["per_mission"][mid]
        print(f"  {mid:<8}{t['n_gt_instances']:>6}{t['recall_pct']:>8.1f}%"
              f"{t['precision_pct']:>10.1f}%{t['median_position_error_m']:>9.2f}m"
              f"{t['median_axis_error_deg']:>9.1f}d")
    write_results("field_perception", res)
    figures()


if __name__ == "__main__":
    main()
