"""GPX file parser for bicycle route analysis.

Reads GPX tracks, extracts lat/lon/elevation data, and computes per-segment
grades (rise/run × 100 %).  Surface type (``paved`` / ``unpaved``) is read
from the ``<type>`` child element of each ``<trk>`` block.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrackSegment:
    """One ``<trkseg>`` block with its parsed track-points and surface label.

    Attributes:
        points:  List of ``(lat, lon, elevation_m)`` tuples.  Elevation may be
                 *None* if the GPX file omits ``<ele>`` tags.
        surface: ``'paved'``, ``'unpaved'``, or *None* when unknown.
    """
    points: List[Tuple[float, float, Optional[float]]]
    surface: Optional[str] = None


@dataclass
class GpxRoute:
    """A parsed GPX route composed of one or more :class:`TrackSegment` objects.

    Attributes:
        name:     Human-readable route name (from ``<name>`` or filename stem).
        segments: Ordered list of track segments.
        filepath: Absolute path to the source GPX file, or *None*.
    """
    name: str
    segments: List[TrackSegment]
    filepath: Optional[str] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def total_distance_m(self) -> float:
        """Return total horizontal route distance in metres."""
        total = 0.0
        for seg in self.segments:
            pts = seg.points
            for i in range(len(pts) - 1):
                total += _haversine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        return total

    def get_grades(
        self,
        split_surface: bool = False,
    ):
        """Return grade values (%) together with the associated segment distances.

        Parameters
        ----------
        split_surface:
            When *False* (default) returns a tuple ``(grades, distances)`` where
            each list has one entry per consecutive point-pair with valid
            elevation data.

            When *True* returns a dict keyed by surface type
            (``'paved'``, ``'unpaved'``, ``'unknown'``), each value being a
            tuple ``(grades, distances)``.

        Returns
        -------
        tuple or dict
            See *split_surface* description above.
        """
        if split_surface:
            data: dict[str, tuple[list, list]] = {
                "paved": ([], []),
                "unpaved": ([], []),
                "unknown": ([], []),
            }
            for seg in self.segments:
                key = seg.surface if seg.surface in ("paved", "unpaved") else "unknown"
                g, d = _segment_grades(seg.points)
                data[key][0].extend(g)
                data[key][1].extend(d)
            return data
        else:
            all_grades: list[float] = []
            all_dists: list[float] = []
            for seg in self.segments:
                g, d = _segment_grades(seg.points)
                all_grades.extend(g)
                all_dists.extend(d)
            return all_grades, all_dists


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 coordinates."""
    R = 6_371_000.0  # mean Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _segment_grades(
    points: List[Tuple[float, float, Optional[float]]],
) -> Tuple[List[float], List[float]]:
    """Compute grade (%) and horizontal distance (m) for consecutive point pairs.

    Point pairs where either elevation is missing, or the horizontal distance
    is negligible (< 0.01 m), are silently skipped.

    Returns
    -------
    grades : list[float]
        Grade in percent (negative = downhill).
    distances : list[float]
        Horizontal distance in metres for each grade sample.
    """
    grades: list[float] = []
    distances: list[float] = []
    for i in range(len(points) - 1):
        lat1, lon1, ele1 = points[i]
        lat2, lon2, ele2 = points[i + 1]
        if ele1 is None or ele2 is None:
            continue
        h = _haversine(lat1, lon1, lat2, lon2)
        if h < 0.01:
            continue
        grades.append((ele2 - ele1) / h * 100.0)
        distances.append(h)
    return grades, distances


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

# Namespaces commonly used by GPX files
_GPX_NS = (
    "http://www.topografix.com/GPX/1/1",
    "http://www.topografix.com/GPX/1/0",
    "",
)

_SURFACE_KEYWORDS = {"paved", "unpaved"}


def load_gpx(filepath: str) -> GpxRoute:
    """Parse a GPX file and return a :class:`GpxRoute`.

    Surface type is inferred from the ``<type>`` child element of each
    ``<trk>``.  Recognised values are ``paved`` and ``unpaved`` (case-
    insensitive).  If absent, or if the value is unrecognised, the surface is
    *None*.

    Parameters
    ----------
    filepath:
        Path to the ``.gpx`` file.

    Returns
    -------
    GpxRoute
    """
    path = Path(filepath)
    tree = ET.parse(path)
    root = tree.getroot()

    # Detect namespace prefix (e.g. "{http://www.topografix.com/GPX/1/1}")
    ns_prefix = ""
    if root.tag.startswith("{"):
        ns_prefix = root.tag.split("}")[0] + "}"

    def _find(element, tag: str):
        return element.find(f"{ns_prefix}{tag}")

    def _findall(element, tag: str):
        return element.findall(f"{ns_prefix}{tag}")

    # Route name: prefer <name> inside <metadata> or first <trk>
    name = path.stem
    for candidate_path in (f".//{ns_prefix}name",):
        elem = root.find(candidate_path)
        if elem is not None and elem.text and elem.text.strip():
            name = elem.text.strip()
            break

    segments: list[TrackSegment] = []

    for trk in _findall(root, "trk"):
        # Surface from track-level <type>
        trk_surface: Optional[str] = None
        type_elem = _find(trk, "type")
        if type_elem is not None and type_elem.text:
            val = type_elem.text.strip().lower()
            if val in _SURFACE_KEYWORDS:
                trk_surface = val

        for trkseg in _findall(trk, "trkseg"):
            points: list[tuple] = []
            for trkpt in _findall(trkseg, "trkpt"):
                try:
                    lat = float(trkpt.get("lat", ""))
                    lon = float(trkpt.get("lon", ""))
                except ValueError:
                    continue
                ele_elem = _find(trkpt, "ele")
                ele = float(ele_elem.text) if (ele_elem is not None and ele_elem.text) else None
                points.append((lat, lon, ele))

            if len(points) >= 2:
                segments.append(TrackSegment(points=points, surface=trk_surface))

    return GpxRoute(name=name, segments=segments, filepath=str(path))
