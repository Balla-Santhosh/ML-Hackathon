"""Folium map generation for matched routes."""
from __future__ import annotations

import folium
import numpy as np

from .network import RoadNetwork

COLORS = {"gps": "#1f4fff", "network": "#9a9a9a", "ref": "#1a9e1a", "clf": "#ff8c00", "hmm": "#d7191c"}


def _arc_lines(net: RoadNetwork, local_ids):
    ll_a = net.proj.to_lonlat(net.A[local_ids])
    ll_b = net.proj.to_lonlat(net.B[local_ids])
    return [[[a[1], a[0]], [b[1], b[0]]] for a, b in zip(ll_a, ll_b)]


def _legend(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};margin-right:6px;vertical-align:middle"></span>{t}</div>'
        for t, c in items)
    html = ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #888;border-radius:6px;font:12px sans-serif">{rows}</div>')
    return folium.Element(html)


def route_map(track_ll, net: RoadNetwork, xy, hmm_local=None, clf_local=None, ref_global=None,
              title="", max_network_arcs=12000, max_points=2500):
    """Interactive map. Layers: road network, GPS fixes, reference route (optional),
    classifier-only matches (optional) and HMM/Viterbi route."""
    lat, lon = track_ll[:, 1], track_ll[:, 0]
    fm = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    legend = []

    # road network near the track
    lo, hi = xy.min(0) - 150, xy.max(0) + 150
    near = np.flatnonzero(((np.minimum(net.A, net.B) <= hi) & (np.maximum(net.A, net.B) >= lo)).all(1))
    if len(near) > max_network_arcs:
        near = near[np.argsort(net.alen[near])[::-1][:max_network_arcs]]
    g = folium.FeatureGroup("Road network", show=True)
    folium.PolyLine(_arc_lines(net, near), color=COLORS["network"], weight=1, opacity=0.6).add_to(g)
    g.add_to(fm); legend.append(("Road network", COLORS["network"]))

    if ref_global is not None:
        loc = [net.g2l[int(a)] for a in ref_global if int(a) in net.g2l]
        g = folium.FeatureGroup("Reference route", show=True)
        folium.PolyLine(_arc_lines(net, np.array(loc, int)), color=COLORS["ref"], weight=7, opacity=0.55).add_to(g)
        g.add_to(fm); legend.append(("Reference route", COLORS["ref"]))

    if clf_local is not None:
        u = np.unique(clf_local)
        g = folium.FeatureGroup("Classifier-only matches", show=False)
        folium.PolyLine(_arc_lines(net, u), color=COLORS["clf"], weight=4, opacity=0.9).add_to(g)
        g.add_to(fm); legend.append(("Classifier-only (toggle)", COLORS["clf"]))

    if hmm_local is not None:
        u = np.unique(hmm_local)
        g = folium.FeatureGroup("Predicted route (classifier + HMM)", show=True)
        folium.PolyLine(_arc_lines(net, u), color=COLORS["hmm"], weight=4, opacity=0.95).add_to(g)
        g.add_to(fm); legend.append(("Predicted route (HMM)", COLORS["hmm"]))

    step = max(1, len(track_ll) // max_points)
    g = folium.FeatureGroup("GPS observations", show=True)
    for i in range(0, len(track_ll), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8, weight=1,
                            tooltip=f"fix {i}, t={track_ll[i, 2]:.0f}s" if track_ll.shape[1] > 2 else None).add_to(g)
    g.add_to(fm); legend.append(("GPS fixes", COLORS["gps"]))
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(fm)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(fm)

    fm.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(fm)
    fm.get_root().html.add_child(_legend(legend))
    if title:
        fm.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 8px;'
            f'border:1px solid #888;border-radius:4px;font:bold 13px sans-serif">{title}</div>'))
    return fm
