"""Reading and cleaning the raw per-trip files.

Raw layout (one folder per trip, files named <trip_id>.<ext>):
  .track  lon lat t          (GPS fixes, ~1 Hz)
  .nodes  lon lat            (row index = node_id)
  .arcs   from_node to_node  (row index = arc_id)
  .route  arc_id             (reference arc sequence; NOT used for prediction)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n,3) lon, lat, t   (cleaned)
    nodes: np.ndarray            # (N,2) lon, lat
    arcs: np.ndarray             # (M,2) from, to
    route: np.ndarray | None = None
    cleaning: dict = field(default_factory=dict)


def _read(path: Path, dtype=float, ncols=None) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.size and arr.shape[1] != ncols:
        raise ValueError(f"{path} expected {ncols} cols, got {arr.shape[1]}")
    return arr


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop non-finite / implausible fixes, sort by time, remove duplicate timestamps."""
    info = {"raw_rows": int(len(track))}
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    ok &= ~((track[:, 0] == 0) & (track[:, 1] == 0))
    info["dropped_invalid"] = int((~ok).sum())
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    info["was_unsorted"] = bool((order != np.arange(len(order))).any())
    track = track[order]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    info["dropped_dup_time"] = int((~keep).sum())
    track = track[keep]
    info["clean_rows"] = int(len(track))
    return track, info


def load_trip(trip_dir: str | Path, with_route: bool = True) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track, info = clean_track(_read(trip_dir / f"{tid}.track", float, 3))
    nodes = _read(trip_dir / f"{tid}.nodes", float, 2)
    arcs = _read(trip_dir / f"{tid}.arcs", np.int64, 2)
    route = None
    rpath = trip_dir / f"{tid}.route"
    if with_route and rpath.exists():
        route = _read(rpath, np.int64).ravel()
    return Trip(tid, track, nodes, arcs, route, info)


def load_route(trip_dir: str | Path) -> np.ndarray:
    """Reference route only (server-side evaluation)."""
    trip_dir = Path(trip_dir)
    return _read(trip_dir / f"{trip_dir.name}.route", np.int64).ravel()


def list_trips(part_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(part_dir).iterdir() if p.is_dir())
