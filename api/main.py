"""PathSnap API (runs on Laptop 1).

Endpoints
  GET  /health                -> status + loaded model
  GET  /trips                 -> the evaluation trips available on this server (ids + #fixes only)
  POST /predict {"trip_id"}   -> runs the full pipeline (candidates -> features -> classifier -> HMM/Viterbi)
                                 and returns the predicted route + server-side accuracy metrics.

Reference routes are read ONLY inside `server_side_evaluation`, after prediction has
finished; they are never passed to the predictor and never returned to the client.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
      (set PATHSNAP_EVAL_DIR to the folder containing the evaluation trips)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import EVAL_DIR  # noqa: E402
from pathsnap.io import list_trips, load_route  # noqa: E402
from pathsnap.labels import align_to_route  # noqa: E402
from pathsnap.metrics import continuity, route_metrics  # noqa: E402
from pathsnap.pipeline import Predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor()
log.info("Loaded model=%s  hmm=%s  eval_dir=%s", predictor.meta["model"], predictor.hmm_params, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def _trip_dirs():
    return {d.name: d for d in list_trips(EVAL_DIR)}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.meta["model"], "hmm_params": predictor.hmm_params.to_dict(),
            "n_eval_trips": len(_trip_dirs())}


@app.get("/trips")
def trips():
    out = []
    for tid, d in _trip_dirs().items():
        with open(d / f"{tid}.track") as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_fixes": n})
    return {"trips": out}


def server_side_evaluation(res: dict, trip_dir: Path) -> dict:
    """Score the prediction against the server's reference route (kept on this machine)."""
    route = load_route(trip_dir)
    net, xy = res["net"], res["xy"]
    gt, _, _ = align_to_route(xy, net.node_xy, res["trip"].arcs, route)
    nxy = net.node_xy
    arcs = res["trip"].arcs
    arc_len = np.linalg.norm(nxy[arcs[:, 1]] - nxy[arcs[:, 0]], axis=1)
    rm = route_metrics(res["hmm_arcs"], route, arc_len)
    return {
        "top1_accuracy_hmm": float(np.mean(res["hmm_arcs"] == gt)),
        "top1_accuracy_classifier_only": float(np.mean(res["clf_arcs"] == gt)),
        "route_continuity_hmm": float(continuity(net, res["hmm_local"], res["hmm_offset"], xy)),
        "route_continuity_classifier_only": float(continuity(net, res["clf_local"], res["clf_offset"], xy)),
        "route_precision_len": rm["route_precision_len"],
        "route_recall_len": rm["route_recall_len"],
        "route_mismatch_fraction": rm["route_mismatch_fraction"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    dirs = _trip_dirs()
    if req.trip_id not in dirs:
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    log.info("Inference request for trip %s", req.trip_id)
    res = predictor.predict_trip(dirs[req.trip_id])
    metrics = server_side_evaluation(res, dirs[req.trip_id])
    net, proj = res["net"], res["proj"]
    track = res["trip"].track

    # geometry for the UI: GPS fixes, predicted route segments, nearby road network
    def seg_ll(local_ids):
        a = proj.to_lonlat(net.A[local_ids]); b = proj.to_lonlat(net.B[local_ids])
        return [[[round(p[1], 7), round(p[0], 7)], [round(q[1], 7), round(q[0], 7)]] for p, q in zip(a, b)]

    route_local = [net.g2l[a] for a in res["route_arcs"]]
    xy = res["xy"]
    lo, hi = xy.min(0) - 120, xy.max(0) + 120
    near = np.flatnonzero(((np.minimum(net.A, net.B) <= hi) & (np.maximum(net.A, net.B) >= lo)).all(1))
    if len(near) > 8000:
        near = near[np.argsort(net.alen[near])[::-1][:8000]]
    clf_only = np.setdiff1d(np.unique(res["clf_local"]), np.unique(res["hmm_local"]))

    log.info("trip %s: %d fixes | top1 HMM=%.4f clf=%.4f | continuity=%.4f | breaks=%d | %.2fs",
             req.trip_id, len(track), metrics["top1_accuracy_hmm"], metrics["top1_accuracy_classifier_only"],
             metrics["route_continuity_hmm"], res["breaks"], res["timings"]["total_s"])
    return {
        "trip_id": req.trip_id,
        "n_fixes": int(len(track)),
        "gps": [[round(float(la), 7), round(float(lo_), 7)] for lo_, la in track[:, :2]],
        "predicted_arc_per_fix": res["hmm_arcs"].tolist(),
        "predicted_route_arcs": [int(a) for a in res["route_arcs"]],
        "predicted_route_segments": seg_ll(np.array(route_local, int)),
        "classifier_only_extra_segments": seg_ll(clf_only),
        "road_network_segments": seg_ll(near),
        "metrics": metrics,
        "hmm_breaks": int(res["breaks"]),
        "timings": {k: round(v, 3) for k, v in res["timings"].items()},
        "model": predictor.meta["model"],
    }
