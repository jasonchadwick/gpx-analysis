"""Tests for gpxanalysis.

Run with::

    pytest tests/
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gpxanalysis import GpxRoute, TrackSegment, GradeVisualizer, load_gpx
from gpxanalysis.parser import _haversine, _segment_grades

# ---------------------------------------------------------------------------
# Paths to test fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "test_data"
FLAT_PAVED = DATA_DIR / "flat_paved.gpx"
HILLY_MIXED = DATA_DIR / "hilly_mixed.gpx"


# ===========================================================================
# parser.py tests
# ===========================================================================

class TestHaversine:
    """_haversine() returns the correct great-circle distance."""

    def test_same_point_is_zero(self):
        assert _haversine(47.6, -122.3, 47.6, -122.3) == pytest.approx(0.0)

    def test_one_degree_latitude(self):
        # 1° latitude ≈ 111 195 m  (varies slightly with latitude)
        d = _haversine(0.0, 0.0, 1.0, 0.0)
        assert 111_000 < d < 111_400

    def test_symmetry(self):
        d1 = _haversine(47.6, -122.3, 47.7, -122.3)
        d2 = _haversine(47.7, -122.3, 47.6, -122.3)
        assert d1 == pytest.approx(d2)


class TestSegmentGrades:
    """_segment_grades() computes correct grades and distances."""

    def test_flat_segment(self):
        # Two points ≈1 km apart, same elevation → grade = 0
        points = [(47.600, -122.3, 100.0), (47.609, -122.3, 100.0)]
        grades, dists = _segment_grades(points)
        assert len(grades) == 1
        assert grades[0] == pytest.approx(0.0, abs=1e-9)
        assert dists[0] == pytest.approx(
            _haversine(47.600, -122.3, 47.609, -122.3), rel=1e-9
        )

    def test_uphill_grade(self):
        # 50 m rise over ≈1 000 m horizontal  → ~5 %
        points = [(47.600, -122.3, 0.0), (47.60901, -122.3, 50.0)]
        grades, dists = _segment_grades(points)
        assert len(grades) == 1
        assert grades[0] == pytest.approx(5.0, abs=0.1)

    def test_downhill_grade(self):
        points = [(47.600, -122.3, 50.0), (47.60901, -122.3, 0.0)]
        grades, dists = _segment_grades(points)
        assert grades[0] == pytest.approx(-5.0, abs=0.1)

    def test_missing_elevation_skipped(self):
        points = [(47.600, -122.3, None), (47.609, -122.3, 50.0)]
        grades, dists = _segment_grades(points)
        assert len(grades) == 0

    def test_duplicate_point_skipped(self):
        points = [(47.600, -122.3, 50.0), (47.600, -122.3, 50.0)]
        grades, dists = _segment_grades(points)
        assert len(grades) == 0

    def test_multiple_points(self):
        points = [
            (47.600, -122.3, 0.0),
            (47.60901, -122.3, 50.0),   # +5 %
            (47.61802, -122.3, 100.0),  # +5 %
        ]
        grades, dists = _segment_grades(points)
        assert len(grades) == 2
        for g in grades:
            assert g == pytest.approx(5.0, abs=0.1)


class TestLoadGpx:
    """load_gpx() correctly parses GPX files."""

    def test_flat_paved_name(self):
        route = load_gpx(FLAT_PAVED)
        assert "Flat Paved Route" in route.name

    def test_flat_paved_filepath(self):
        route = load_gpx(FLAT_PAVED)
        assert route.filepath is not None
        assert route.filepath.endswith(".gpx")

    def test_flat_paved_one_segment(self):
        route = load_gpx(FLAT_PAVED)
        assert len(route.segments) == 1

    def test_flat_paved_surface(self):
        route = load_gpx(FLAT_PAVED)
        assert route.segments[0].surface == "paved"

    def test_flat_paved_four_points(self):
        route = load_gpx(FLAT_PAVED)
        assert len(route.segments[0].points) == 4

    def test_flat_paved_grades_near_zero(self):
        route = load_gpx(FLAT_PAVED)
        grades, _ = route.get_grades()
        assert all(abs(g) < 0.01 for g in grades)

    def test_hilly_mixed_two_tracks(self):
        route = load_gpx(HILLY_MIXED)
        assert len(route.segments) == 2

    def test_hilly_mixed_surfaces(self):
        route = load_gpx(HILLY_MIXED)
        surfaces = {seg.surface for seg in route.segments}
        assert "paved" in surfaces
        assert "unpaved" in surfaces

    def test_hilly_mixed_grades_paved_positive(self):
        route = load_gpx(HILLY_MIXED)
        data = route.get_grades(split_surface=True)
        paved_grades = data["paved"][0]
        assert len(paved_grades) > 0
        assert all(g > 0 for g in paved_grades)

    def test_hilly_mixed_grades_unpaved_negative(self):
        route = load_gpx(HILLY_MIXED)
        data = route.get_grades(split_surface=True)
        unpaved_grades = data["unpaved"][0]
        assert len(unpaved_grades) > 0
        assert all(g < 0 for g in unpaved_grades)

    def test_hilly_mixed_grade_magnitude(self):
        route = load_gpx(HILLY_MIXED)
        grades, _ = route.get_grades()
        magnitudes = [abs(g) for g in grades]
        # Paved grades ≈ 4.5–6 %, unpaved grades ≈ 7.5–8 %
        for m in magnitudes:
            assert 4.0 < m < 9.0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_gpx("/nonexistent/path/route.gpx")


class TestGpxRouteHelpers:
    """GpxRoute helper methods."""

    def test_total_distance_flat_route(self):
        route = load_gpx(FLAT_PAVED)
        # 3 segments × ~1 000 m each ≈ 3 000 m
        total = route.total_distance_m()
        assert 2_800 < total < 3_200

    def test_get_grades_returns_lists(self):
        route = load_gpx(FLAT_PAVED)
        grades, dists = route.get_grades()
        assert isinstance(grades, list)
        assert isinstance(dists, list)
        assert len(grades) == len(dists)

    def test_get_grades_split_surface_keys(self):
        route = load_gpx(HILLY_MIXED)
        data = route.get_grades(split_surface=True)
        assert set(data.keys()) == {"paved", "unpaved", "unknown"}


# ===========================================================================
# grade_visualizer.py tests
# ===========================================================================

@pytest.fixture()
def flat_route():
    return load_gpx(FLAT_PAVED)


@pytest.fixture()
def hilly_route():
    return load_gpx(HILLY_MIXED)


@pytest.fixture()
def viz_with_routes(flat_route, hilly_route):
    v = GradeVisualizer()
    v.add_route(flat_route, label="Flat")
    v.add_route(hilly_route, label="Hilly")
    return v


class TestGradeVisualizerSetup:
    def test_add_route_default_label(self, flat_route):
        v = GradeVisualizer()
        v.add_route(flat_route)
        assert v._labels[0] == flat_route.name

    def test_add_route_custom_label(self, flat_route):
        v = GradeVisualizer()
        v.add_route(flat_route, label="My Route")
        assert v._labels[0] == "My Route"

    def test_clear(self, viz_with_routes):
        viz_with_routes.clear()
        assert viz_with_routes._routes == []
        assert viz_with_routes._labels == []


class TestPlotPdf:
    def test_returns_axes(self, viz_with_routes):
        ax = viz_with_routes.plot_pdf(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_axes_xlabel(self, viz_with_routes):
        ax = viz_with_routes.plot_pdf(show=False)
        assert "grade" in ax.get_xlabel().lower()
        plt.close("all")

    def test_axes_title_contains_pdf(self, viz_with_routes):
        ax = viz_with_routes.plot_pdf(show=False)
        assert "PDF" in ax.get_title() or "pdf" in ax.get_title().lower()
        plt.close("all")

    def test_normalize_changes_ylabel(self, viz_with_routes):
        ax_norm = viz_with_routes.plot_pdf(normalize=True, show=False)
        ax_raw = viz_with_routes.plot_pdf(normalize=False, show=False)
        assert ax_norm.get_ylabel() != ax_raw.get_ylabel()
        plt.close("all")

    def test_split_surface_produces_more_lines(self, hilly_route):
        v = GradeVisualizer()
        v.add_route(hilly_route, label="Hilly")
        ax_split = v.plot_pdf(split_surface=True, show=False)
        ax_merged = v.plot_pdf(split_surface=False, show=False)
        assert len(ax_split.lines) > len(ax_merged.lines)
        plt.close("all")

    def test_custom_axes_used(self, viz_with_routes):
        fig, ax = plt.subplots()
        returned = viz_with_routes.plot_pdf(ax=ax, show=False)
        assert returned is ax
        plt.close("all")

    def test_grade_range_clipping(self, viz_with_routes):
        ax = viz_with_routes.plot_pdf(grade_range=(-5.0, 5.0), show=False)
        xlim = ax.get_xlim()
        # The plot should be roughly within the clipped range
        assert xlim[0] < 0 and xlim[1] > 0
        plt.close("all")

    def test_empty_visualizer_no_error(self):
        v = GradeVisualizer()
        ax = v.plot_pdf(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestPlotCdf:
    def test_returns_axes(self, viz_with_routes):
        ax = viz_with_routes.plot_cdf(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_axes_title_contains_cdf(self, viz_with_routes):
        ax = viz_with_routes.plot_cdf(show=False)
        assert "CDF" in ax.get_title() or "cdf" in ax.get_title().lower()
        plt.close("all")

    def test_normalize_ylabel_probability(self, viz_with_routes):
        ax = viz_with_routes.plot_cdf(normalize=True, show=False)
        assert "probability" in ax.get_ylabel().lower()
        plt.close("all")

    def test_unnormalized_ylabel_distance(self, viz_with_routes):
        ax = viz_with_routes.plot_cdf(normalize=False, show=False)
        assert "distance" in ax.get_ylabel().lower()
        plt.close("all")

    def test_custom_axes_used(self, viz_with_routes):
        fig, ax = plt.subplots()
        returned = viz_with_routes.plot_cdf(ax=ax, show=False)
        assert returned is ax
        plt.close("all")


class TestPlotCombined:
    def test_returns_figure_and_two_axes(self, viz_with_routes):
        fig, (ax_pdf, ax_cdf) = viz_with_routes.plot(show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax_pdf, plt.Axes)
        assert isinstance(ax_cdf, plt.Axes)
        plt.close("all")

    def test_combined_both_axes_labelled(self, viz_with_routes):
        _, (ax_pdf, ax_cdf) = viz_with_routes.plot(show=False)
        assert "PDF" in ax_pdf.get_title() or "pdf" in ax_pdf.get_title().lower()
        assert "CDF" in ax_cdf.get_title() or "cdf" in ax_cdf.get_title().lower()
        plt.close("all")


class TestNormalizationBehaviour:
    """Verify that normalize=True produces sensible outputs."""

    def test_cdf_normalized_max_is_one(self, hilly_route):
        v = GradeVisualizer()
        v.add_route(hilly_route, label="Hilly")
        ax = v.plot_cdf(normalize=True, show=False)
        # The last y-value of the step line should equal 1.0
        line = ax.lines[0]  # first line (zero-grade vline is not a step line here)
        # Find step-function lines (they won't be the vertical dashed line)
        step_lines = [l for l in ax.lines if l.get_linestyle() != "--"]
        assert len(step_lines) > 0
        ydata = step_lines[0].get_ydata()
        assert ydata[-1] == pytest.approx(1.0, abs=1e-6)
        plt.close("all")

    def test_distance_weighted_pdf_integrates_to_one(self, hilly_route):
        """Distance-weighted KDE should still integrate to ≈ 1."""
        from scipy.stats import gaussian_kde
        grades, dists = hilly_route.get_grades()
        weights = np.array(dists) / np.sum(dists)
        kde = gaussian_kde(grades, weights=weights)
        x = np.linspace(-30, 30, 2000)
        dx = x[1] - x[0]
        integral = np.trapezoid(kde(x), x) if hasattr(np, "trapezoid") else np.trapz(kde(x), x)
        assert integral == pytest.approx(1.0, abs=0.02)
