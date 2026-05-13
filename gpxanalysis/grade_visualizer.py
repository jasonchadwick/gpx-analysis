"""Grade distribution visualizer for GPX bicycle routes.

Provides :class:`GradeVisualizer`, which accumulates one or more
:class:`~gpxanalysis.parser.GpxRoute` objects and renders their grade
distributions as probability density functions (PDF) and/or cumulative
distribution functions (CDF).

Typical usage::

    from gpxanalysis import load_gpx, GradeVisualizer

    viz = GradeVisualizer()
    viz.add_route(load_gpx("gpx/route1.gpx"), label="Morning loop")
    viz.add_route(load_gpx("gpx/route2.gpx"), label="Evening loop")
    viz.plot(normalize=True, split_surface=True)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from .parser import GpxRoute


# ---------------------------------------------------------------------------
# Colour cycling helper
# ---------------------------------------------------------------------------

_BASE_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

_SURFACE_STYLE = {
    "paved":   dict(linestyle="-"),
    "unpaved": dict(linestyle="--"),
    "unknown": dict(linestyle=":"),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GradeVisualizer:
    """Accumulate routes and visualize their grade distributions.

    Routes are added with :meth:`add_route` and then rendered with
    :meth:`plot_pdf`, :meth:`plot_cdf`, or :meth:`plot` (both side-by-side).

    Parameters
    ----------
    bw_method:
        Bandwidth method passed to :class:`scipy.stats.gaussian_kde`.
        Accepts ``'scott'`` (default), ``'silverman'``, or a scalar.
    """

    def __init__(self, bw_method: str | float = "scott") -> None:
        self._routes: List[GpxRoute] = []
        self._labels: List[str] = []
        self._bw_method = bw_method

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def add_route(self, route: GpxRoute, label: Optional[str] = None) -> None:
        """Add a route to the visualizer.

        Parameters
        ----------
        route:
            A parsed :class:`~gpxanalysis.parser.GpxRoute`.
        label:
            Display name; defaults to ``route.name``.
        """
        self._routes.append(route)
        self._labels.append(label if label is not None else route.name)

    def clear(self) -> None:
        """Remove all routes."""
        self._routes.clear()
        self._labels.clear()

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _colour(self, index: int) -> str:
        return _BASE_COLOURS[index % len(_BASE_COLOURS)]

    def _build_kde(
        self,
        grades: list[float],
        distances: list[float],
        normalize: bool,
        x: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fit a KDE and evaluate it on *x*.

        When *normalize* is *True* the KDE is weighted by segment distance so
        that the density represents *fraction of distance* at each grade, rather
        than fraction of raw samples.

        Returns *None* if fewer than two finite grade samples exist.
        """
        arr = np.array(grades, dtype=float)
        finite_mask = np.isfinite(arr)
        arr = arr[finite_mask]
        if len(arr) < 2:
            return None

        weights = None
        if normalize:
            dists = np.array(distances, dtype=float)[finite_mask]
            total = dists.sum()
            if total > 0:
                weights = dists / total

        if np.std(arr) < 1e-10:
            # All grades are identical – KDE covariance is singular.
            # Add a tiny symmetric jitter so we get a narrow spike instead
            # of an error.  The noise is reproducible and negligibly small.
            noise = float(np.abs(np.mean(arr))) * 1e-3 or 1e-3
            rng = np.random.default_rng(0)
            arr = arr + rng.normal(0.0, noise, size=arr.shape)

        try:
            kde = gaussian_kde(arr, bw_method=self._bw_method, weights=weights)
        except np.linalg.LinAlgError:
            return None
        return kde(x)

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def plot_pdf(
        self,
        normalize: bool = False,
        split_surface: bool = False,
        grade_range: Tuple[float, float] = (-30.0, 30.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Axes:
        """Plot the grade probability density function (PDF) for all routes.

        Parameters
        ----------
        normalize:
            When *True*, each grade sample is weighted by its segment's
            horizontal distance so that the density reflects the fraction of
            *distance* spent at each grade level (rather than fraction of
            segments).  Useful for fairly comparing routes of different lengths.
        split_surface:
            When *True*, paved and unpaved sub-distributions are plotted with
            different line styles on the same axes.
        grade_range:
            ``(min_grade, max_grade)`` range for the x-axis (%).
        ax:
            Existing :class:`matplotlib.axes.Axes` to draw on.  A new figure
            is created when *None*.
        show:
            Call :func:`matplotlib.pyplot.show` when done.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        x = np.linspace(grade_range[0], grade_range[1], 500)

        for idx, (route, label) in enumerate(zip(self._routes, self._labels)):
            colour = self._colour(idx)
            if split_surface:
                data = route.get_grades(split_surface=True)
                first = True
                for surface, (grades, dists) in data.items():
                    if len(grades) < 2:
                        continue
                    y = self._build_kde(grades, dists, normalize, x)
                    if y is None:
                        continue
                    style = _SURFACE_STYLE.get(surface, {})
                    lbl = f"{label} ({surface})" if not first else f"{label} ({surface})"
                    ax.plot(x, y, color=colour, label=lbl, **style)
                    first = False
            else:
                grades, dists = route.get_grades(split_surface=False)
                if len(grades) < 2:
                    continue
                y = self._build_kde(grades, dists, normalize, x)
                if y is None:
                    continue
                ax.plot(x, y, color=colour, label=label)

        ax.set_xlabel("Grade (%)")
        ax.set_ylabel("Distance-weighted density" if normalize else "Density")
        ax.set_title("Grade Distribution – PDF")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)
        if self._routes:
            ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_cdf(
        self,
        normalize: bool = False,
        split_surface: bool = False,
        grade_range: Tuple[float, float] = (-30.0, 30.0),
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Axes:
        """Plot the cumulative distribution function (CDF) of grades.

        Parameters
        ----------
        normalize:
            When *True*, the y-axis shows cumulative probability (0 – 1).
            When *False*, the y-axis shows distance-weighted cumulative sum
            (metres), showing the actual amount of route at or below each grade.
        split_surface:
            When *True*, paved and unpaved sub-distributions are plotted with
            different line styles on the same axes.
        grade_range:
            ``(min_grade, max_grade)`` clipping range for the x-axis (%).
        ax:
            Existing :class:`matplotlib.axes.Axes` to draw on.
        show:
            Call :func:`matplotlib.pyplot.show` when done.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        for idx, (route, label) in enumerate(zip(self._routes, self._labels)):
            colour = self._colour(idx)
            if split_surface:
                data = route.get_grades(split_surface=True)
                for surface, (grades, dists) in data.items():
                    if not grades:
                        continue
                    style = _SURFACE_STYLE.get(surface, {})
                    lbl = f"{label} ({surface})"
                    self._draw_cdf(ax, grades, dists, normalize, grade_range,
                                   color=colour, label=lbl, **style)
            else:
                grades, dists = route.get_grades(split_surface=False)
                if not grades:
                    continue
                self._draw_cdf(ax, grades, dists, normalize, grade_range,
                               color=colour, label=label)

        ax.set_xlabel("Grade (%)")
        ax.set_ylabel("Cumulative probability" if normalize else "Cumulative distance (m)")
        ax.set_title("Grade Distribution – CDF")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)
        if self._routes:
            ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot(
        self,
        normalize: bool = False,
        split_surface: bool = False,
        grade_range: Tuple[float, float] = (-30.0, 30.0),
        show: bool = True,
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """Plot PDF and CDF side-by-side in a single figure.

        Parameters
        ----------
        normalize:
            Passed to both :meth:`plot_pdf` and :meth:`plot_cdf`.
        split_surface:
            Passed to both :meth:`plot_pdf` and :meth:`plot_cdf`.
        grade_range:
            ``(min_grade, max_grade)`` range for both plots.
        show:
            Call :func:`matplotlib.pyplot.show` when done.

        Returns
        -------
        fig : Figure
        axes : tuple[Axes, Axes]
            ``(ax_pdf, ax_cdf)``
        """
        fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(18, 5))
        self.plot_pdf(
            normalize=normalize,
            split_surface=split_surface,
            grade_range=grade_range,
            ax=ax_pdf,
            show=False,
        )
        self.plot_cdf(
            normalize=normalize,
            split_surface=split_surface,
            grade_range=grade_range,
            ax=ax_cdf,
            show=False,
        )
        if show:
            plt.tight_layout()
            plt.show()
        return fig, (ax_pdf, ax_cdf)

    # ------------------------------------------------------------------
    # Internal CDF helper
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_cdf(
        ax: Axes,
        grades: list[float],
        distances: list[float],
        normalize: bool,
        grade_range: Tuple[float, float],
        **plot_kwargs,
    ) -> None:
        """Sort grades and draw a weighted step-CDF on *ax*."""
        arr = np.array(grades, dtype=float)
        dists = np.array(distances, dtype=float)

        # Clip and sort
        clipped = np.clip(arr, grade_range[0], grade_range[1])
        order = np.argsort(clipped)
        clipped = clipped[order]
        dists = dists[order]

        cumulative = np.cumsum(dists)
        if normalize and cumulative[-1] > 0:
            cumulative = cumulative / cumulative[-1]

        ax.step(clipped, cumulative, where="post", **plot_kwargs)
