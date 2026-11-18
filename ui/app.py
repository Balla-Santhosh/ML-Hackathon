"""PathSnap evaluator UI (runs on Laptop 2).

Run:  streamlit run ui/app.py -- --api http://<laptop1-ip>:8000
  or  set PATHSNAP_API_URL=http://<laptop1-ip>:8000 && streamlit run ui/app.py

The UI only talks to the API: it lists the evaluation trips, sends the selected
trip_id to POST /predict and draws what comes back. It never sees reference routes.
"""
import argparse
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

parser = argparse.ArgumentParser()
parser.add_argument("--api", default=os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
args, _ = parser.parse_known_args()

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - GPS map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", args.api).rstrip("/")
try:
    health = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"API online - model: {health['model']}")
    st.sidebar.caption(f"HMM params: {health['hmm_params']}")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api}: {e}")
    st.stop()

trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
labels = {f"Trip {t['trip_id']}  ({t['n_fixes']} GPS fixes)": t["trip_id"] for t in trips}
st.sidebar.subheader(f"Evaluation trips ({len(trips)})")
choice = st.sidebar.radio("Select a trip", list(labels))
run = st.sidebar.button("Reconstruct route", type="primary", use_container_width=True)

if run:
    with st.spinner(f"Sending trip {labels[choice]} to the API ..."):
        r = requests.post(f"{api}/predict", json={"trip_id": labels[choice]}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()

res = st.session_state.get("result")
if not res:
    st.info("Select one of the evaluation trips on the left and press **Reconstruct route**.")
    st.stop()

m = res["metrics"]
st.subheader(f"Trip {res['trip_id']} - {res['n_fixes']} GPS fixes - model {res['model']} + HMM/Viterbi")
c = st.columns(5)
c[0].metric("Top-1 accuracy (HMM)", f"{m['top1_accuracy_hmm']:.2%}",
            f"{m['top1_accuracy_hmm'] - m['top1_accuracy_classifier_only']:+.2%} vs classifier-only")
c[1].metric("Top-1 classifier-only", f"{m['top1_accuracy_classifier_only']:.2%}")
c[2].metric("Route continuity (HMM)", f"{m['route_continuity_hmm']:.2%}",
            f"{m['route_continuity_hmm'] - m['route_continuity_classifier_only']:+.2%}")
c[3].metric("Route precision / recall (length)", f"{m['route_precision_len']:.2%} / {m['route_recall_len']:.2%}")
c[4].metric("Processing time", f"{res['timings']['total_s']:.2f} s", f"{res['hmm_breaks']} HMM breaks", delta_color="off")

gps = res["gps"]
lats = [p[0] for p in gps]; lons = [p[1] for p in gps]
fm = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=14, control_scale=True)
g = folium.FeatureGroup("Road network")
folium.PolyLine(res["road_network_segments"], color="#9a9a9a", weight=1, opacity=0.6).add_to(g); g.add_to(fm)
g = folium.FeatureGroup("Classifier-only (segments HMM rejected)", show=False)
if res["classifier_only_extra_segments"]:
    folium.PolyLine(res["classifier_only_extra_segments"], color="#ff8c00", weight=4).add_to(g)
g.add_to(fm)
g = folium.FeatureGroup("Predicted route (classifier + HMM)")
folium.PolyLine(res["predicted_route_segments"], color="#d7191c", weight=5, opacity=0.9).add_to(g); g.add_to(fm)
g = folium.FeatureGroup("GPS observations")
step = max(1, len(gps) // 2500)
for i in range(0, len(gps), step):
    folium.CircleMarker(gps[i], radius=2, color="#1f4fff", fill=True, weight=1, tooltip=f"fix {i}").add_to(g)
g.add_to(fm)
folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green")).add_to(fm)
folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red")).add_to(fm)
fm.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
folium.LayerControl(collapsed=False).add_to(fm)
fm.get_root().html.add_child(folium.Element(
    '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:6px 10px;border:1px solid #888;'
    'border-radius:6px;font:12px sans-serif">'
    '<div><span style="display:inline-block;width:16px;height:4px;background:#d7191c"></span> Predicted route</div>'
    '<div><span style="display:inline-block;width:16px;height:4px;background:#ff8c00"></span> Classifier-only extras</div>'
    '<div><span style="display:inline-block;width:8px;height:8px;border-radius:4px;background:#1f4fff"></span> GPS fixes</div>'
    '<div><span style="display:inline-block;width:16px;height:2px;background:#9a9a9a"></span> Road network</div></div>'))
components.html(fm.get_root().render(), height=650)

with st.expander("Raw API response details"):
    st.json({k: v for k, v in res.items() if k not in ("gps", "road_network_segments", "predicted_route_segments",
                                                       "classifier_only_extra_segments", "predicted_arc_per_fix")})
    st.write(f"Predicted route: {len(res['predicted_route_arcs'])} distinct consecutive arcs")
