# gpx-analysis

A Python toolkit for analysing planned GPX bicycle routes.

---

## Features

| Feature | Details |
|---|---|
| **Grade visualiser** | Plot the distribution of road grades (rise ÷ run × 100 %) along any route |
| **PDF & CDF** | Probability density function and cumulative distribution function, side-by-side or separately |
| **Multiple routes** | Overlay several routes on the same axes for direct comparison |
| **Normalisation** | Distance-weighted density (PDF) and cumulative-probability (CDF) so routes of different lengths are comparable |
| **Paved / unpaved split** | Render paved and unpaved sections of a route as separate distributions (solid vs. dashed lines) |

---

## Directory layout

```
gpx-analysis/
├── gpx/                    # ← drop your .gpx files here
├── gpxanalysis/            # main package
│   ├── __init__.py
│   ├── parser.py           # GPX parsing & grade calculation
│   └── grade_visualizer.py # GradeVisualizer class
├── tests/
│   ├── test_data/          # small synthetic GPX files used by the test suite
│   │   ├── flat_paved.gpx
│   │   └── hilly_mixed.gpx
│   └── test_grade_visualizer.py
├── requirements.txt
├── pyproject.toml
└── README.md               # this file
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # installs gpxanalysis as an editable package
```

---

## Quick-start

### 1 — Load a route and inspect grades

```python
from gpxanalysis import load_gpx

route = load_gpx("gpx/my_commute.gpx")

print(route.name)
print(f"Total distance: {route.total_distance_m() / 1000:.1f} km")

grades, distances = route.get_grades()
print(f"Steepest climb: {max(grades):.1f} %")
print(f"Steepest descent: {min(grades):.1f} %")
```

### 2 — Plot a single route

```python
from gpxanalysis import load_gpx, GradeVisualizer

route = load_gpx("gpx/my_commute.gpx")

viz = GradeVisualizer()
viz.add_route(route, label="My commute")
viz.plot()          # shows PDF and CDF side-by-side
```

### 3 — Compare multiple routes

```python
from gpxanalysis import load_gpx, GradeVisualizer

viz = GradeVisualizer()
viz.add_route(load_gpx("gpx/route_a.gpx"), label="Route A")
viz.add_route(load_gpx("gpx/route_b.gpx"), label="Route B")
viz.add_route(load_gpx("gpx/route_c.gpx"), label="Route C")

# Normalised comparison (distance-weighted PDF, probability CDF)
viz.plot(normalize=True)
```

### 4 — PDF and CDF separately

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
viz.plot_pdf(normalize=True, ax=axes[0], show=False)
viz.plot_cdf(normalize=True, ax=axes[1], show=False)
plt.tight_layout()
plt.savefig("grade_distributions.png", dpi=150)
plt.show()
```

### 5 — Paved vs. unpaved split

Mark surface type in the `<type>` child of each `<trk>` element in your GPX
file:

```xml
<trk>
  <name>Gravel climb</name>
  <type>unpaved</type>
  <trkseg>…</trkseg>
</trk>
```

Accepted values: `paved`, `unpaved` (case-insensitive).  Tracks without a
recognised `<type>` are labelled *unknown*.

```python
viz.plot(split_surface=True)     # solid = paved, dashed = unpaved
```

---

## API reference

### `load_gpx(filepath) → GpxRoute`

Parse a `.gpx` file and return a `GpxRoute`.

### `GpxRoute`

| Attribute / method | Description |
|---|---|
| `.name` | Route name (from `<name>` tag or filename stem) |
| `.segments` | List of `TrackSegment` objects |
| `.filepath` | Absolute path to the source file |
| `.total_distance_m()` | Total horizontal distance in metres |
| `.get_grades(split_surface=False)` | Returns `(grades, distances)` or a dict keyed by surface |

### `GradeVisualizer`

| Method | Description |
|---|---|
| `.add_route(route, label=None)` | Add a `GpxRoute` |
| `.clear()` | Remove all routes |
| `.plot_pdf(normalize, split_surface, grade_range, ax, show)` | PDF plot |
| `.plot_cdf(normalize, split_surface, grade_range, ax, show)` | CDF plot |
| `.plot(normalize, split_surface, grade_range, show)` | Both plots side-by-side |

All plot methods accept an optional `ax` argument so you can embed them in
larger figures, and `show=False` to suppress the automatic `plt.show()` call.

#### `normalize` parameter

| Plot | `normalize=False` | `normalize=True` |
|---|---|---|
| PDF | Density weighted equally per GPS segment | Density weighted by segment horizontal distance |
| CDF | Cumulative distance (m) on y-axis | Cumulative probability [0, 1] on y-axis |

---

## Running the tests

```bash
pytest tests/ -v
```

---

## Supported GPX format

The parser uses Python's built-in `xml.etree.ElementTree` — no third-party
GPX library is required.  It supports:

* GPX 1.0 and GPX 1.1 (auto-detects namespace)
* Multiple `<trk>` blocks per file (treated as separate segments)
* `<ele>` elevation tags on `<trkpt>` points
* `<type>` tag on `<trk>` for surface labelling (`paved` / `unpaved`)

Point-pairs that lack elevation data, or whose haversine distance is less than
0.01 m, are silently skipped.