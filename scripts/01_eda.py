"""Exploratory data analysis of the raw dataset (train + student_test partitions).

Outputs (results/eda/ and results/figures/):
  inventory.csv           per-trip file sizes, row counts, gaps, extent, network stats
  data_quality.csv        missing / invalid values per file type
  figures/eda_*.png       distributions, trajectory + road-network plots
  maps/eda_trip_<id>.html Folium map (GPS, road network, reference route)
"""
import sys
from pathlib import Path

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import FIG_DIR, MAP_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.geo import LocalProjection, haversine  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402

EDA_DIR = RESULTS_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)


def raw_quality(path, ncols, dtype=float):
    """Inspect a raw file without cleaning: rows, non-numeric / NaN / wrong-width lines."""
    rows = bad_width = nonnum = 0
    for line in open(path):
        parts = line.split()
        rows += 1
        if len(parts) != ncols:
            bad_width += 1
            continue
        try:
            vals = [dtype(x) for x in parts]
            if any(v != v for v in vals):
                nonnum += 1
        except ValueError:
            nonnum += 1
    return rows, bad_width, nonnum


def trip_stats(d: Path, part: str):
    tid = d.name
    rec = {"partition": part, "trip_id": tid}
    quality = []
    for ext, nc, dt in [("track", 3, float), ("nodes", 2, float), ("arcs", 2, int), ("route", 1, int)]:
        f = d / f"{tid}.{ext}"
        rec[f"{ext}_exists"] = f.exists()
        rec[f"{ext}_kb"] = f.stat().st_size / 1024 if f.exists() else np.nan
        if f.exists():
            r, bw, nn = raw_quality(f, nc, dt)
            quality.append({"partition": part, "trip_id": tid, "file": ext, "rows": r, "bad_width": bw, "non_numeric_or_nan": nn})
    tr = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int, ndmin=2)
    route = np.loadtxt(d / f"{tid}.route", dtype=int, ndmin=1)
    dt = np.diff(tr[:, 2])
    step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    speed = step / np.maximum(dt, 1e-9)
    rec.update({
        "n_obs": len(tr), "duration_s": tr[-1, 2] - tr[0, 2],
        "dt_median": np.median(dt), "dt_max": dt.max(), "frac_dt_1s": np.mean(dt == 1),
        "n_gaps_gt5s": int((dt > 5).sum()), "n_nonincreasing_t": int((dt <= 0).sum()),
        "track_len_km": step.sum() / 1000, "speed_p95_mps": np.percentile(speed, 95),
        "n_speed_gt70mps": int((speed > 70).sum()),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(), "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "extent_km": haversine(tr[:, 0].min(), tr[:, 1].min(), tr[:, 0].max(), tr[:, 1].max()) / 1000,
        "n_nodes": len(nodes), "n_arcs": len(arcs), "route_len_arcs": len(route),
    })
    # network connectivity
    N = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(N, N))
    rec["weak_components"] = connected_components(g, directed=True, connection="weak")[0]
    ns, lab = connected_components(g, directed=True, connection="strong")
    rec["strong_components"] = ns
    rec["largest_scc_frac"] = np.bincount(lab).max() / N
    deg = np.bincount(arcs.ravel(), minlength=N)
    rec["isolated_nodes"] = int((deg == 0).sum())
    pairs = set(map(tuple, arcs))
    rec["bidirectional_frac"] = np.mean([(b, a) in pairs for a, b in arcs[:5000]])
    rec["self_loops"] = int((arcs[:, 0] == arcs[:, 1]).sum())
    rec["route_contiguous_frac"] = float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) if len(route) > 1 else 1.0
    rec["route_repeats"] = len(route) - len(set(route))
    proj = LocalProjection.for_track(tr)
    nxy = proj.to_xy(nodes[:, 0], nodes[:, 1])
    rec["arc_len_median_m"] = float(np.median(np.linalg.norm(nxy[arcs[:, 1]] - nxy[arcs[:, 0]], axis=1)))
    return rec, quality, dt


def plot_trip(ax, d: Path, title):
    tid = d.name
    tr = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int, ndmin=2)
    route = np.loadtxt(d / f"{tid}.route", dtype=int, ndmin=1)
    lo, hi = tr[:, :2].min(0) - 0.003, tr[:, :2].max(0) + 0.003
    A, B = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    m = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([A[m], B[m]], 1), colors="0.8", linewidths=0.4))
    ax.add_collection(LineCollection(np.stack([nodes[arcs[route, 0]], nodes[arcs[route, 1]]], 1), colors="green", linewidths=1.6))
    ax.scatter(tr[:, 0], tr[:, 1], s=1, c="blue")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def folium_trip(d: Path, out: Path):
    tid = d.name
    tr = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int, ndmin=2)
    route = np.loadtxt(d / f"{tid}.route", dtype=int, ndmin=1)
    lo, hi = tr[:, :2].min(0) - 0.002, tr[:, :2].max(0) + 0.002
    A, B = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    m = np.flatnonzero(((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1))[:15000]
    fm = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network (arcs)")
    for a in m:
        folium.PolyLine([[A[a, 1], A[a, 0]], [B[a, 1], B[a, 0]]], color="#999999", weight=1, opacity=0.6).add_to(net)
    net.add_to(fm)
    ref = folium.FeatureGroup("Reference route (.route)")
    for a in route:
        folium.PolyLine([[A[a, 1], A[a, 0]], [B[a, 1], B[a, 0]]], color="green", weight=4).add_to(ref)
    ref.add_to(fm)
    gps = folium.FeatureGroup("Raw GPS fixes (.track)")
    for lon, lat, t in tr[:: max(1, len(tr) // 1500)]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, popup=f"t={t:.0f}s").add_to(gps)
    gps.add_to(fm)
    fm.fit_bounds([[lo[1], lo[0]], [hi[1], hi[0]]])
    folium.LayerControl(collapsed=False).add_to(fm)
    fm.save(out)


def main():
    recs, quals, dts = [], [], []
    for part, pdir in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for d in list_trips(pdir):
            r, q, dt = trip_stats(d, part)
            recs.append(r); quals += q; dts.append(pd.DataFrame({"partition": part, "dt": dt}))
    inv = pd.DataFrame(recs)
    inv.to_csv(EDA_DIR / "inventory.csv", index=False)
    qual = pd.DataFrame(quals)
    qual.to_csv(EDA_DIR / "data_quality.csv", index=False)
    dts = pd.concat(dts)

    summary = inv.groupby("partition").agg(
        trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
        obs_max=("n_obs", "max"), dur_median_s=("duration_s", "median"), len_km_median=("track_len_km", "median"),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
        trips_with_gap_gt5s=("n_gaps_gt5s", lambda s: int((s > 0).sum())), weak_comp_median=("weak_components", "median"),
        isolated_nodes_total=("isolated_nodes", "sum"), route_contig_min=("route_contiguous_frac", "min"),
    )
    summary.to_csv(EDA_DIR / "partition_summary.csv")
    print(summary.T)
    print("\nMissing/invalid values per file type:\n", qual.groupby(["partition", "file"])[["rows", "bad_width", "non_numeric_or_nan"]].sum())
    print("\nSampling interval share == 1 s:", (dts.dt == 1).mean().round(4), " max gap:", dts.dt.max())

    # --- figures ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = inv[inv.partition == part]
        ax[0, 0].hist(s.n_obs, bins=30, alpha=.6, label=part, color=c)
        ax[0, 1].hist(s.track_len_km, bins=30, alpha=.6, label=part, color=c)
        ax[0, 2].hist(np.log10(s.n_arcs), bins=30, alpha=.6, label=part, color=c)
        ax[1, 1].hist(s.speed_p95_mps, bins=30, alpha=.6, label=part, color=c)
    ax[0, 0].set_title("GPS fixes per trip"); ax[0, 1].set_title("Track length (km)")
    ax[0, 2].set_title("log10(#arcs) in trip network"); ax[1, 1].set_title("95th pct speed (m/s)")
    d = dts.dt.clip(upper=12)
    ax[1, 0].hist(d, bins=np.arange(0, 13.5, 0.5), log=True, color="gray"); ax[1, 0].set_title("Sampling interval (s, clipped at 12, log)")
    ax[1, 2].scatter((inv.lon_min + inv.lon_max) / 2, (inv.lat_min + inv.lat_max) / 2,
                     c=inv.partition.map({"train": "tab:blue", "student_test": "tab:orange"}), s=15)
    ax[1, 2].set_title("Trip locations (lon, lat)")
    for a in ax.flat[:5]:
        a.legend(fontsize=7) if a.get_legend_handles_labels()[0] else None
    fig.suptitle("PathSnap EDA - train (80) vs student_test (10)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_distributions.png", dpi=130); plt.close(fig)

    # missing-value bar chart
    q = qual.groupby("file")[["bad_width", "non_numeric_or_nan"]].sum()
    ax = q.plot.bar(figsize=(6, 3.5), title="Missing / malformed rows per file type (all 90 trips)")
    ax.set_ylabel("rows"); plt.tight_layout(); plt.savefig(FIG_DIR / "eda_missing_values.png", dpi=130); plt.close()

    # trajectories with road network
    sample = list_trips(TRAIN_DIR)[:6] + list_trips(STUDENT_TEST_DIR)[:2]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for a, dd in zip(axs.flat, sample):
        plot_trip(a, dd, f"{dd.parent.name}/{dd.name}")
    fig.suptitle("GPS fixes (blue), reference route (green), road network (grey)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_trajectories.png", dpi=130); plt.close(fig)

    # label alignment quality (needs 02_build_dataset to have run)
    obs_p = Path(RESULTS_DIR.parent / "data_cache" / "obs_train.parquet")
    if obs_p.exists():
        o = pd.read_parquet(obs_p)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].hist(o.align_dist.clip(upper=200), bins=80, log=True); ax[0].axvline(30, color="r", ls="--")
        ax[0].set_title("train: distance GPS fix -> aligned reference arc (m)")
        per = o.groupby("trip_id").align_dist.median().sort_values()
        ax[1].bar(range(len(per)), per.values); ax[1].set_yscale("log"); ax[1].set_title("train: median alignment distance per trip")
        fig.tight_layout(); fig.savefig(FIG_DIR / "eda_label_alignment.png", dpi=130); plt.close(fig)

    folium_trip(TRAIN_DIR / "00000001", MAP_DIR / "eda_trip_train_00000001.html")
    folium_trip(STUDENT_TEST_DIR / "00000007", MAP_DIR / "eda_trip_student_test_00000007.html")
    print("saved EDA outputs to", EDA_DIR, FIG_DIR, MAP_DIR)


if __name__ == "__main__":
    main()
