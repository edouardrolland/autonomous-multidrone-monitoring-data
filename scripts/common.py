"""Shared paths, plotting style and small helpers for the reproduction scripts.

Every script in this directory reads only from ../data/ and writes only to ../outputs/.
Nothing here depends on the original research repository.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SIM = DATA / "simulation"
FIELD = DATA / "field"
META = DATA / "metadata"
OUT = ROOT / "outputs"
FIGDIR = OUT / "figures"

MISSIONS = ["F1", "F2", "F3", "F4"]
SURFACES = ["dorsal", "left_flank", "right_flank"]
REGIMES = ["grazing", "travelling", "panic"]
DELTA_TOST = 0.005           # equivalence margin of the validation campaign
MATCH_GATE_M = 15.0          # detection-to-label association gate
TAU = 0.5                    # conventional coverage level

PAPER_STYLE = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.linewidth": 0.6,
}

C_SURFACE = {"dorsal": "#009E73", "left_flank": "#0072B2", "right_flank": "#D55E00"}
C_REGIME = {"grazing": "#2e7d32", "travelling": "#1565c0", "panic": "#c62828",
            "generalist": "#616161"}


def use_paper_style():
    plt.rcParams.update(PAPER_STYLE)


def ensure_outputs():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return OUT


def load(name: str) -> pd.DataFrame:
    """Load a dataset by bare filename, from whichever data subdirectory holds it."""
    for d in (SIM, FIELD, META):
        p = d / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{name} not found under {DATA}")


def save_fig(fig, name: str):
    ensure_outputs()
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path.relative_to(ROOT)}")
    return path


def write_results(section: str, results: dict):
    """Append one section's reproduced numbers to outputs/statistics.json."""
    ensure_outputs()
    path = OUT / "statistics.json"
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob[section] = results
    path.write_text(json.dumps(blob, indent=1, default=_jsonable))
    print(f"  [stats]  {path.relative_to(ROOT)}  <- {section}")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# --------------------------------------------------------------------------- geometry
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Same formula the deployed evaluation used."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def axis_error_deg(a, b):
    """Undirected body-axis difference, folded to [0, 90] deg."""
    d = np.abs((a - b) % 180.0)
    return np.minimum(d, 180.0 - d)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, input order preserved."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def tost(diffs, delta=DELTA_TOST):
    """Two one-sided paired t-tests for equivalence within +-delta. Returns (p, equivalent)."""
    from scipy import stats
    x = np.asarray(diffs, dtype=float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return (0.0, True) if abs(x.mean()) < delta else (1.0, False)
    t_lo = (x.mean() + delta) / se
    t_hi = (x.mean() - delta) / se
    p = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    return float(p), bool(p < 0.05)


def boot_ci(x, stat=np.median, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    d = np.array([stat(rng.choice(x, len(x), True)) for _ in range(n)])
    return float(stat(x)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
