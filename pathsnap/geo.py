"""Geometry helpers.

Trips are spread across the globe but each trip covers at most a few tens of km,
so we use a local equirectangular projection centred on the trip
(x = R*cos(lat0)*dlon, y = R*dlat). Over 50 km the scale error is well below 1 %,
which is far smaller than GPS noise, and it lets us use fast Euclidean maths in metres.
"""
import numpy as np

EARTH_R = 6_371_008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R

    @classmethod
    def for_track(cls, track):
        return cls(np.median(track[:, 0]), np.median(track[:, 1]))

    def to_xy(self, lon, lat):
        return np.c_[(np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky]

    def to_lonlat(self, xy):
        xy = np.asarray(xy)
        return np.c_[xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0]


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(P, A, B):
    """Distance from points P to segments AB (broadcasting, last dim = 2).

    Returns (distance, t) with t in [0,1] the clamped projection parameter.
    """
    AB = B - A
    L2 = (AB ** 2).sum(-1)
    t = ((P - A) * AB).sum(-1) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[..., None] * AB
    return np.sqrt(((P - proj) ** 2).sum(-1)), t


def bearing(dx, dy):
    """Planar bearing in radians, 0 = north, clockwise."""
    return np.arctan2(dx, dy)
