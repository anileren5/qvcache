#!/usr/bin/env python3
"""Bar figures of hit ratio and hit recall for paper plots.

Each PDF is one dataset. Each panel is one workload: grouped bars for
QVCache and Aker, with hit ratio and hit recall side by side. The two
10M search figures also show end-of-stream average latency, in
milliseconds, on a separate right axis. That latency is not the measured
wall time: QVCache runs were overloaded. Backend latency is the median
Aker interval latency at hit ratio 0, and each bar is
hit ratio * hit latency + (1 - hit ratio) * (backend + miss penalty).
Hit latency and miss penalty are drawn under each panel on their own
millisecond scale. Refresh figures use the evaluation phase only.
Search figures use the full stream.

Outputs, under figures/ by default:

- spacev-1m-refresh.pdf
- spacev-1m-refresh-rows12.pdf
- spacev-1m-refresh-row3.pdf
- sphere-1m-refresh.pdf
- spacev-10m-search.pdf
- spacev-10m-search-dev0.05-rows12.pdf
- spacev-10m-search-dev0.05-row3.pdf
- sphere-10m-search.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SYSTEMS = ("qvcache", "aker")
SYSTEM_COLOR = {
    "qvcache": "#D55E00",
    "qvcache-b4": "#CC79A7",
    "qvcache-dev0.01": "#332288",
    "qvcache-dev0.05": "#009E73",
    "qvcache-dev0.25": "#6A3D9A",
    "aker": "#0072B2",
}
SYSTEM_NAME = {
    "qvcache": "QVCache",
    "qvcache-b4": "QVCache b4",
    "qvcache-dev0.01": "QVCache 0.01",
    "qvcache-dev0.05": "QVCache 0.05",
    "qvcache-dev0.25": "QVCache 0.25",
    "aker": "Aker",
}
AKER_EXACT = "#F0C808"
AKER_APPROX = SYSTEM_COLOR["aker"]
SIMZIPF = r"simZipf ($\lambda = U(0, 0.5)$)"
METRICS = ("hit_ratio", "hit_recall")
METRIC_LABELS = ("Hit ratio", "Hit recall")


class Summary:
    """Final rates for one system. Ratios are fractions in [0, 1]."""

    def __init__(
        self,
        hit_ratio: float,
        hit_recall: float,
        n: int,
        exact_ratio: float | None = None,
        approx_ratio: float | None = None,
        avg_latency_ms: float | None = None,
        hit_latency_ms: float | None = None,
        miss_penalty_ms: float | None = None,
    ):
        self.hit_ratio = hit_ratio
        self.hit_recall = hit_recall
        self.n = n
        self.exact_ratio = exact_ratio
        self.approx_ratio = approx_ratio
        self.avg_latency_ms = avg_latency_ms
        self.hit_latency_ms = hit_latency_ms
        self.miss_penalty_ms = miss_penalty_ms


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "hatch.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def iter_events(path: Path):
    with path.open() as handle:
        for line in handle:
            start = line.find('{"event"')
            if start < 0:
                continue
            try:
                yield json.loads(line[start:])
            except json.JSONDecodeError:
                continue


def search_summary(path: Path) -> Summary:
    """End-of-stream hit ratio, hit recall, and average latency.

    Hit recall is the hit-weighted mean of interval recall_cache_hits, which
    matches cum_recall_hits on the refresh logs.     Aker also logs exact and
    approximate hits per interval; those counts sum to the hit ratio.
    Hit latency and miss penalty are weighted by the interval hit and miss
    counts. avg_latency_ms stays the measured cumulative mean; the search
    figures replace it with the reconstructed value.
    """
    recall_mass = 0.0
    hit_count = 0
    exact = 0
    approx = 0
    has_split = False
    hit_lat_mass = 0.0
    hit_n = 0
    penalty_mass = 0.0
    miss_n = 0
    last = None
    for event in iter_events(path):
        if event.get("event") != "stream_metrics":
            continue
        hits = event.get("cache_hit_count")
        recall = event.get("recall_cache_hits")
        if hits and recall is not None:
            recall_mass += float(recall) * int(hits)
            hit_count += int(hits)
        interval_hits = int(event.get("hits") or 0)
        interval_n = int(event.get("interval_queries") or 0)
        interval_misses = interval_n - interval_hits
        if interval_hits and event.get("avg_hit_latency_ms") is not None:
            hit_lat_mass += float(event["avg_hit_latency_ms"]) * interval_hits
            hit_n += interval_hits
        if interval_misses > 0 and event.get("miss_penalty_ms") is not None:
            penalty_mass += float(event["miss_penalty_ms"]) * interval_misses
            miss_n += interval_misses
        if "exact_hits" in event and "approx_hits" in event:
            has_split = True
            exact += int(event["exact_hits"])
            approx += int(event["approx_hits"])
        last = event
    if last is None:
        raise SystemExit(f"no stream_metrics events in {path}")
    n = int(last["completed"])
    return Summary(
        hit_ratio=float(last["cum_hit_ratio"]),
        hit_recall=recall_mass / hit_count if hit_count else float("nan"),
        n=n,
        exact_ratio=(exact / n) if has_split and n else None,
        approx_ratio=(approx / n) if has_split and n else None,
        avg_latency_ms=float(last["cum_avg_latency_ms"]),
        hit_latency_ms=(hit_lat_mass / hit_n) if hit_n else 0.0,
        miss_penalty_ms=(penalty_mass / miss_n) if miss_n else 0.0,
    )


def refresh_eval_summary(path: Path) -> Summary:
    """Hit ratio and hit recall at the end of the refresh evaluation phase."""
    last = None
    for event in iter_events(path):
        if event.get("event") == "refresh_progress" and event.get("phase") == "eval":
            last = event
    if last is None:
        raise SystemExit(f"no eval-phase refresh_progress events in {path}")
    return Summary(
        hit_ratio=float(last["cum_hit_ratio"]),
        hit_recall=float(last["cum_recall_hits"]),
        n=int(last["completed"]),
    )


def _aker_zero_pairs(path: Path) -> list[tuple[float, float]]:
    """Latency and recall on intervals that never hit the cache."""
    pairs = []
    for event in iter_events(path):
        if event.get("event") != "stream_metrics":
            continue
        if float(event.get("hit_ratio", 1.0)) != 0.0 or event.get("recall_all") is None:
            continue
        pairs.append((float(event["avg_latency_ms"]), float(event["recall_all"])))
    return pairs


def aker_nocache_reference(run_dir: Path) -> tuple[float, float]:
    """Median backend latency (ms) and recall with no cache hits.

    Workloads with no such interval use the median over every Aker
    hit-ratio-0 interval in the dataset.
    """
    own = run_dir / "aker.log"
    pairs = _aker_zero_pairs(own) if own.is_file() else []
    if not pairs:
        for path in sorted(run_dir.parent.glob("*/aker.log")):
            pairs.extend(_aker_zero_pairs(path))
    if not pairs:
        raise SystemExit(f"no Aker hit-ratio 0 intervals for {run_dir}")
    latencies, recalls = zip(*pairs)
    return statistics.median(latencies), statistics.median(recalls)


def aker_backend_ms(run_dir: Path) -> float:
    """Median Aker latency on intervals with hit ratio 0."""
    return aker_nocache_reference(run_dir)[0]


def search_latency_peak(spec: dict) -> float:
    """Largest reconstructed average latency across a search figure."""
    systems = spec.get("systems", SYSTEMS)
    peak = 0.0
    for row_key, _label in spec["rows"]:
        for col_key, _col_label in spec["cols"]:
            run_dir = spec["root"] / row_key[col_key]
            if not run_dir.is_dir():
                continue
            backend_ms = aker_backend_ms(run_dir)
            for row in load_panel(run_dir, "search", systems).values():
                peak = max(peak, reconstructed_latency_ms(row, backend_ms))
    return peak


def reconstructed_latency_ms(row: Summary, backend_ms: float) -> float:
    """hit ratio * hit latency + (1 - hit ratio) * (backend + miss penalty)."""
    hit_ratio = row.hit_ratio
    hit_latency = row.hit_latency_ms or 0.0
    penalty = row.miss_penalty_ms or 0.0
    return hit_ratio * hit_latency + (1.0 - hit_ratio) * (backend_ms + penalty)


def load_panel(run_dir: Path, kind: str, systems: tuple[str, ...]) -> dict[str, Summary]:
    loader = refresh_eval_summary if kind == "refresh" else search_summary
    series = {}
    for system in systems:
        path = run_dir / f"{system}.log"
        if path.is_file():
            series[system] = loader(path)
    if not series:
        raise SystemExit(f"no logs for {', '.join(systems)} in {run_dir}")
    return series


def bar_style(system: str, metric: str, hatch_recall: bool = True) -> dict:
    """Color identifies the system. Black hatch marks hit recall."""
    color = SYSTEM_COLOR[system]
    if metric == "hit_ratio" or not hatch_recall:
        return {"facecolor": color, "edgecolor": color, "linewidth": 0.4, "hatch": None}
    return {"facecolor": color, "edgecolor": "black", "linewidth": 0.4, "hatch": "///"}


def legend_handles(
    split_aker: bool,
    systems: tuple[str, ...],
    names: dict[str, str] | None = None,
    compact: bool = False,
) -> list[Patch]:
    names = names or SYSTEM_NAME
    handles = []
    for system in systems:
        name = names.get(system, SYSTEM_NAME[system])
        if system == "aker" and split_aker:
            if compact:
                handles.append(Patch(label=name, **bar_style("aker", "hit_ratio")))
            else:
                handles.append(Patch(facecolor=AKER_EXACT, edgecolor=AKER_EXACT, linewidth=0.4, label="Aker exact"))
                handles.append(Patch(facecolor=AKER_APPROX, edgecolor=AKER_APPROX, linewidth=0.4, label="Aker approx."))
                handles.append(Patch(label="Aker hit recall", **bar_style("aker", "hit_recall")))
            continue
        if compact:
            handles.append(Patch(label=name, **bar_style(system, "hit_ratio")))
            continue
        handles.append(Patch(label=f"{name} hit ratio", **bar_style(system, "hit_ratio")))
        handles.append(Patch(label=f"{name} hit recall", **bar_style(system, "hit_recall")))
    return handles


def annotate_bars(
    ax,
    grouped: list[list[tuple]],
    label_size: float,
    pad: float = 0.7,
    text_h: float = 6.0,
    fmt: str = "{:.1f}",
) -> None:
    """Horizontal values just above each bar.

    A pair whose numbers would collide lifts only the taller bar's label, and
    only by enough to clear the other number.
    """
    for group in grouped:
        labels = []
        for bar, height in group:
            labels.append([bar.get_x() + bar.get_width() / 2, height + pad, height])
        ordered = sorted(labels, key=lambda item: item[2])
        for index, label in enumerate(ordered):
            for other in ordered[:index]:
                if abs(label[0] - other[0]) < 0.28 and label[1] < other[1] + text_h:
                    label[1] = other[1] + text_h
        for x, y, height in labels:
            ax.text(
                x,
                y,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=label_size,
                color="#1c1917",
                zorder=4,
                clip_on=False,
            )


def latency_scale(max_ms: float) -> tuple[float, list[float]]:
    """Right-axis range shared by every panel, with room for the bar labels."""
    padded = max_ms * 1.22
    step = 50.0
    for candidate in (5, 10, 20, 25, 50):
        if padded / candidate <= 6:
            step = float(candidate)
            break
    ylim = math.ceil(padded / step) * step
    ticks = [i * step for i in range(int(round(ylim / step)))]
    return ylim, ticks


def _group_span(center: float, offsets: list[float], width: float) -> tuple[float, float]:
    pad = 0.06
    if not offsets:
        return center - width / 2 - pad, center + width / 2 + pad
    return center + min(offsets) - width / 2 - pad, center + max(offsets) + width / 2 + pad


def _nocache_line(ax, x0: float, x1: float, y: float) -> None:
    ax.plot(
        [x0, x1],
        [y, y],
        color="#1c1917",
        linestyle=(0, (3, 1.4)),
        linewidth=0.9,
        solid_capstyle="butt",
        zorder=5,
        clip_on=False,
    )


def draw_panel(
    ax,
    series: dict,
    label_size: float,
    systems: tuple[str, ...],
    show_latency: bool = False,
    hatch_recall: bool = True,
    nocache: tuple[float, float] | None = None,
):
    present = [system for system in systems if system in series]
    if len(present) >= 6:
        width, step = 0.12, 0.145
    elif len(present) >= 4:
        width, step = 0.16, 0.18
    elif len(present) >= 3:
        width, step = 0.22, 0.24
    elif len(present) > 1:
        width, step = 0.32, 0.42
    else:
        width, step = 0.46, 0.0
    offsets = [(i - (len(present) - 1) / 2) * step for i in range(len(present))]
    # Latency sits further right so its labels clear the hit-recall numbers.
    centers = [0.0, 1.0, 2.55] if show_latency else [0.0, 1.0]
    x = centers[: len(METRICS)]
    grouped: list[list[tuple]] = [[] for _ in METRICS]
    for system, offset in zip(present, offsets):
        summary = series[system]
        heights = (summary.hit_ratio * 100.0, summary.hit_recall * 100.0)
        split = summary.exact_ratio is not None and summary.approx_ratio is not None
        for metric_i, (metric, height) in enumerate(zip(METRICS, heights)):
            xi = x[metric_i] + offset
            if metric == "hit_ratio" and split:
                exact = summary.exact_ratio * 100.0
                approx = summary.approx_ratio * 100.0
                ax.bar(
                    [xi],
                    [exact],
                    width=width,
                    facecolor=AKER_EXACT,
                    edgecolor=AKER_EXACT,
                    linewidth=0.4,
                    zorder=3,
                )
                top = ax.bar(
                    [xi],
                    [approx],
                    bottom=[exact],
                    width=width,
                    facecolor=AKER_APPROX,
                    edgecolor=AKER_APPROX,
                    linewidth=0.4,
                    zorder=3,
                )
                grouped[metric_i].append((top[0], exact + approx))
                continue
            bars = ax.bar([xi], [height], width=width, zorder=3)
            style = bar_style(system, metric, hatch_recall=hatch_recall)
            bars[0].set_facecolor(style["facecolor"])
            bars[0].set_edgecolor(style["edgecolor"])
            bars[0].set_linewidth(style["linewidth"])
            bars[0].set_hatch(style["hatch"])
            grouped[metric_i].append((bars[0], height))
    annotate_bars(ax, grouped, label_size)
    right = 3.15 if show_latency else len(METRICS) - 0.38
    ax.set_xlim(-0.62, right)
    ax.set_ylim(0, 124)
    tick_labels = list(METRIC_LABELS) + (["Average latency"] if show_latency else [])
    ax.set_xticks(centers[: len(tick_labels)], tick_labels)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="#E4E4E7", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", top=False, right=False, direction="out", labelbottom=True)
    ax.spines["right"].set_visible(False)

    latency_ax = None
    if show_latency:
        latency_ax = ax.twinx()
        latency_ax.spines["top"].set_visible(False)
        latency_ax.spines["left"].set_visible(False)
        latency_ax.tick_params(axis="y", which="both", direction="out", top=False, left=False, width=0.6, size=2.6)
        latency_ax.tick_params(axis="x", bottom=False, labelbottom=False, top=False)
        latency_ax.spines["bottom"].set_visible(False)
        group = []
        for system, offset in zip(present, offsets):
            height = series[system].avg_latency_ms
            xi = centers[len(METRICS)] + offset
            bars = latency_ax.bar([xi], [height], width=width, zorder=3)
            style = bar_style(system, "hit_ratio")
            bars[0].set_facecolor(style["facecolor"])
            bars[0].set_edgecolor(style["edgecolor"])
            bars[0].set_linewidth(style["linewidth"])
            group.append((bars[0], height))
        latency_ax._latency_group = group
    if nocache is not None:
        latency_ms, recall = nocache
        x0, x1 = _group_span(centers[1], offsets, width)
        _nocache_line(ax, x0, x1, recall * 100.0)
        if latency_ax is not None:
            x0, x1 = _group_span(centers[len(METRICS)], offsets, width)
            _nocache_line(latency_ax, x0, x1, latency_ms)
    return latency_ax


def draw_cost_panel(ax, series: dict, systems: tuple[str, ...]) -> list[list[tuple]]:
    """Hit latency and miss penalty. Both are milliseconds, solid system colors."""
    present = [system for system in systems if system in series]
    width, step = (0.32, 0.42) if len(present) > 1 else (0.46, 0.0)
    offsets = [(i - (len(present) - 1) / 2) * step for i in range(len(present))]
    centers = [0.0, 1.2]
    grouped: list[list[tuple]] = [[], []]
    for system, offset in zip(present, offsets):
        summary = series[system]
        for metric_i, height in enumerate((summary.hit_latency_ms or 0.0, summary.miss_penalty_ms or 0.0)):
            bars = ax.bar([centers[metric_i] + offset], [height], width=width, zorder=3)
            style = bar_style(system, "hit_ratio")
            bars[0].set_facecolor(style["facecolor"])
            bars[0].set_edgecolor(style["edgecolor"])
            bars[0].set_linewidth(style["linewidth"])
            grouped[metric_i].append((bars[0], height))
    ax.set_xlim(-0.7, 1.9)
    ax.set_xticks(centers, ["Hit latency", "Miss penalty"])
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="#E4E4E7", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", top=False, right=False, direction="out", labelbottom=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return grouped


def small_ms_scale(max_ms: float) -> tuple[float, list[float]]:
    """Axis for hit latency and miss penalty, both well under a few milliseconds."""
    padded = max(max_ms * 1.55, 0.2)
    step = 1.0
    for candidate in (0.1, 0.2, 0.25, 0.5, 1.0):
        if padded / candidate <= 6:
            step = candidate
            break
    ylim = math.ceil(padded / step) * step
    ticks = [round(i * step, 2) for i in range(int(round(ylim / step)))]
    return float(ylim), ticks


def spread_overlapping_labels(fig) -> None:
    """Lift a label by the minimum that clears another label drawn on top of it.

    Hit recall and latency use different y axes, so a check in data coordinates
    cannot see that their numbers occupy the same spot on the page.
    """
    fig.canvas.draw()
    texts = [text for ax in fig.axes for text in ax.texts]
    for _ in range(12):
        renderer = fig.canvas.get_renderer()
        moved = False
        order = sorted(texts, key=lambda text: text.get_window_extent(renderer).y0)
        for index, lower in enumerate(order):
            lower_box = lower.get_window_extent(renderer)
            for upper in order[index + 1 :]:
                upper_box = upper.get_window_extent(renderer)
                if upper_box.x0 >= lower_box.x1 or upper_box.x1 <= lower_box.x0:
                    continue
                gap = upper_box.y0 - lower_box.y1
                if gap >= 0.8:
                    continue
                dy_px = 0.8 - gap
                axis = upper.axes
                origin = axis.transData.inverted().transform((0, 0))
                shifted = axis.transData.inverted().transform((0, dy_px))
                x, y = upper.get_position()
                upper.set_position((x, y + shifted[1] - origin[1]))
                moved = True
                break
            if moved:
                break
        if not moved:
            break
        fig.canvas.draw()


def panel_letter(index: int) -> str:
    return chr(ord("a") + index)


def render(spec: dict, output: Path) -> None:
    rows = spec["rows"]
    cols = spec["cols"]
    kind = spec["kind"]
    root = spec["root"]
    n_rows = len(rows)
    n_cols = len(cols)
    # Full-width conference figure. Height grows with the workload rows.
    systems = spec.get("systems", SYSTEMS)
    fig_w = spec.get("fig_w", 7.16)
    show_latency = bool(spec.get("latency"))
    # The lower strip is hit latency and miss penalty. Average latency can
    # be drawn without it.
    show_cost = show_latency and spec.get("latency_detail", True)
    if show_cost:
        fig_h = spec.get("fig_h", 0.7 + 2.9 * n_rows)
    else:
        fig_h = spec.get("fig_h", 0.55 + 2.15 * n_rows)
    label_size = 5.0 if len(systems) >= 4 else (5.5 if len(systems) >= 3 else (6.0 if n_cols >= 4 else 6.5))
    cost_axes = None
    if show_cost:
        fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
        grid = fig.add_gridspec(n_rows * 2, n_cols, height_ratios=[2.45, 1.2] * n_rows)
        axes = []
        cost_axes = []
        main_share = None
        cost_share = None
        for r in range(n_rows):
            row_axes = []
            row_cost = []
            for c in range(n_cols):
                ax = fig.add_subplot(grid[r * 2, c], sharex=main_share, sharey=main_share)
                cost = fig.add_subplot(grid[r * 2 + 1, c], sharey=cost_share)
                if main_share is None:
                    main_share = ax
                if cost_share is None:
                    cost_share = cost
                row_axes.append(ax)
                row_cost.append(cost)
            axes.append(row_axes)
            cost_axes.append(row_cost)
    else:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, wspace=0.03, hspace=0.05)

    panel = int(spec.get("panel_start", 0))
    summaries = []
    split_aker = False
    latency_axes = []
    max_latency = 0.0
    cost_groups = []
    max_cost = 0.0
    for r, (row_key, row_label) in enumerate(rows):
        for c, (col_key, col_label) in enumerate(cols):
            ax = axes[r][c]
            run_dir = root / row_key / col_key if kind == "refresh" else root / row_key[col_key]
            letter = f"({panel_letter(panel)}) "
            title = f"{letter}{row_label}\n{col_label}" if r"\lambda" in str(row_label) else f"{letter}{row_label}, {col_label}"
            panel += 1
            cost_ax = cost_axes[r][c] if cost_axes is not None else None
            if not run_dir.is_dir():
                ax.axis("off")
                if cost_ax is not None:
                    cost_ax.axis("off")
                print(f"missing {run_dir}", file=sys.stderr)
                continue
            series = load_panel(run_dir, kind, systems)
            backend_ms = None
            nocache = None
            if show_latency:
                backend_ms, nocache_recall = aker_nocache_reference(run_dir)
                for row in series.values():
                    row.avg_latency_ms = reconstructed_latency_ms(row, backend_ms)
                if spec.get("nocache_lines"):
                    nocache = (backend_ms, nocache_recall)
            split_aker = split_aker or any(row.exact_ratio is not None for row in series.values())
            latency_ax = draw_panel(
                ax,
                series,
                label_size,
                systems,
                show_latency=show_latency,
                hatch_recall=spec.get("hatch_recall", True),
                nocache=nocache,
            )
            if "ylim" in spec:
                ax.set_ylim(0, spec["ylim"])
            if latency_ax is not None:
                latency_axes.append((latency_ax, c == n_cols - 1))
                max_latency = max(max_latency, *(height for _bar, height in latency_ax._latency_group))
                if nocache is not None:
                    max_latency = max(max_latency, nocache[0])
            if cost_ax is not None:
                grouped = draw_cost_panel(cost_ax, series, systems)
                cost_groups.append((cost_ax, grouped, c == 0))
                max_cost = max(max_cost, *(height for group in grouped for _bar, height in group))
                cost_ax.set_xlabel(title)
            else:
                ax.set_xlabel(title)
            for system, row in series.items():
                extra = ""
                if row.exact_ratio is not None and row.approx_ratio is not None:
                    extra = f"  exact {row.exact_ratio:.4f}  approx {row.approx_ratio:.4f}"
                latency = ""
                if row.avg_latency_ms is not None and show_latency:
                    latency = (
                        f"  latency {row.avg_latency_ms:.2f} ms"
                        f"  hit latency {row.hit_latency_ms:.3f} ms"
                        f"  miss penalty {row.miss_penalty_ms:.3f} ms"
                        f"  backend {backend_ms:.2f} ms"
                    )
                summaries.append(
                    f"{spec['name']} {row_label} {col_label} {SYSTEM_NAME[system]}: "
                    f"hit ratio {row.hit_ratio:.4f}  hit recall {row.hit_recall:.4f}{extra}{latency}  "
                    f"n={row.n}"
                )
            if c == 0:
                ax.set_ylabel("Percent")

    if latency_axes:
        max_latency = max(max_latency, float(spec.get("latency_at_least", 0.0)))
        ylim, ticks = latency_scale(max_latency)
        for latency_ax, label_right in latency_axes:
            latency_ax.set_ylim(0, ylim)
            latency_ax.set_yticks(ticks)
            annotate_bars(
                latency_ax,
                [latency_ax._latency_group],
                label_size,
                pad=ylim * 0.012,
                text_h=ylim * 0.055,
            )
            if label_right:
                latency_ax.set_ylabel("ms")
            else:
                latency_ax.tick_params(labelright=False)
                latency_ax.set_yticklabels([])
        if cost_groups:
            ylim, ticks = small_ms_scale(max_cost)
            for cost_ax, grouped, label_left in cost_groups:
                cost_ax.set_ylim(0, ylim)
                cost_ax.set_yticks(ticks)
                annotate_bars(
                    cost_ax,
                    grouped,
                    label_size,
                    pad=ylim * 0.03,
                    text_h=ylim * 0.11,
                    fmt="{:.2f}",
                )
                if label_left:
                    cost_ax.set_ylabel("ms")
                else:
                    cost_ax.tick_params(labelleft=False)
        spread_overlapping_labels(fig)

    handles = legend_handles(
        split_aker,
        systems,
        spec.get("system_names"),
        compact=bool(spec.get("compact_legend")),
    )
    if spec.get("nocache_lines"):
        handles.append(
            Line2D(
                [0],
                [0],
                color="#1c1917",
                linestyle=(0, (3, 1.4)),
                linewidth=0.9,
                label="No cache",
            )
        )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=spec.get("legend_ncol", 4 if len(systems) >= 3 else (5 if split_aker else 4)),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    for line in summaries:
        print(line)


def figures(root: Path) -> list[dict]:
    skews = (
        ("0.99", r"$\alpha$ = 0.99"),
        ("0.6", r"$\alpha$ = 0.6"),
        ("0.3", r"$\alpha$ = 0.3"),
        ("uniform", "uniform"),
    )
    deletes = (
        ("0.05", "ins 0.05, del 0.05"),
        ("0.15", "ins 0.05, del 0.15"),
        ("0.25", "ins 0.05, del 0.25"),
    )

    def search_rows(families: list[tuple[str, str, str]]) -> list[tuple[dict[str, str], str]]:
        rows = []
        for label, prefix, suffix in families:
            mapping = {}
            for key, _col_label in skews:
                stem = "uniform" if key == "uniform" else key
                mapping[key] = f"{prefix}-{stem}{suffix}"
            rows.append((mapping, label))
        return rows

    sphere_search_rows = search_rows(
        [
            ("simZipf", "sim-100k", ""),
            ("simZipf+", "simplus-100k", ""),
        ]
    )
    spacev_search_rows = search_rows(
        [
            (SIMZIPF, "sim-100k", ""),
            (r"simZipf+ ($\lambda = 0.01$)", "simplus-100k", "-lam0.01"),
            (r"simZipf+ ($\lambda = U(0, 0.5)$)", "simplus-100k", "-lamU0-0.5"),
        ]
    )

    def refresh_spec(
        name: str,
        dataset: str,
        families: list[tuple[str, str]],
        delete_prefix: str,
        panel_start: int = 0,
    ) -> dict:
        return {
            "name": name,
            "kind": "refresh",
            "root": root / "logs/refresh" / dataset,
            "rows": families,
            "cols": [(f"{delete_prefix}{rate}", label) for rate, label in deletes],
            "compact_legend": True,
            "hatch_recall": False,
            "legend_ncol": 3,
            "systems": ("qvcache", "qvcache-dev0.05", "aker"),
            "system_names": {
                "qvcache": r"QVCache ($D = 0$, default)",
                "qvcache-dev0.05": r"QVCache ($D=0.05$)",
                "aker": "Aker",
            },
            "fig_w": 7.6,
            "panel_start": panel_start,
            "output": f"{name}.pdf",
        }

    spacev_refresh_rows = [
        ("sim-100k-0.99", SIMZIPF),
        ("simplus-100k-0.99-lamU0-0.5", r"simZipf+ ($\lambda = U(0, 0.5)$)"),
        ("simplus-100k-0.99-lam0.01", r"simZipf+ ($\lambda = 0.01$)"),
    ]
    specs = [
        refresh_spec(
            "spacev-1m-refresh",
            "spacev-1m",
            spacev_refresh_rows,
            "ins0.05-del",
        ),
        refresh_spec(
            "spacev-1m-refresh-rows12",
            "spacev-1m",
            spacev_refresh_rows[:2],
            "ins0.05-del",
        ),
        refresh_spec(
            "spacev-1m-refresh-row3",
            "spacev-1m",
            spacev_refresh_rows[2:],
            "ins0.05-del",
            panel_start=6,
        ),
        refresh_spec(
            "sphere-1m-refresh",
            "sphere-1m",
            [
                ("sim-100k-0.99", "simZipf"),
                ("simplus-100k-0.99", "simZipf+"),
            ],
            "del",
        ),
        {
            "name": "spacev-10m-search",
            "kind": "search",
            "root": root / "logs/search/spacev-10m",
            "rows": spacev_search_rows,
            "cols": list(skews),
            "latency": True,
            "fig_w": 7.6,
            "output": "spacev-10m-search.pdf",
        },
        {
            "name": "spacev-10m-search-dev0.05",
            "kind": "search",
            "root": root / "logs/search/spacev-10m",
            "rows": [spacev_search_rows[0], spacev_search_rows[2], spacev_search_rows[1]],
            "cols": list(skews),
            "systems": ("qvcache", "qvcache-dev0.05", "aker"),
            "system_names": {
                "qvcache": r"QVCache (default, $D=0$)",
                "qvcache-dev0.05": r"QVCache ($D=0.05$)",
                "aker": "Aker",
            },
            "compact_legend": True,
            "hatch_recall": False,
            "legend_ncol": 3,
            "latency": True,
            "latency_detail": False,
            "fig_w": 7.6,
            "output": "spacev-10m-search-dev0.05.pdf",
        },
        {
            "name": "spacev-10m-search-dev0.05-rows12",
            "kind": "search",
            "root": root / "logs/search/spacev-10m",
            "rows": [spacev_search_rows[0], spacev_search_rows[2]],
            "cols": list(skews),
            "systems": ("qvcache", "qvcache-dev0.05", "aker"),
            "system_names": {
                "qvcache": r"QVCache (default, $D=0$)",
                "qvcache-dev0.05": r"QVCache ($D=0.05$)",
                "aker": "Aker",
            },
            "compact_legend": True,
            "hatch_recall": False,
            "legend_ncol": 3,
            "latency": True,
            "latency_detail": False,
            "fig_w": 7.6,
            "panel_start": 0,
            "output": "spacev-10m-search-dev0.05-rows12.pdf",
        },
        {
            "name": "spacev-10m-search-dev0.05-row3",
            "kind": "search",
            "root": root / "logs/search/spacev-10m",
            "rows": [spacev_search_rows[1]],
            "cols": list(skews),
            "systems": ("qvcache", "qvcache-dev0.05", "aker"),
            "system_names": {
                "qvcache": r"QVCache (default, $D=0$)",
                "qvcache-dev0.05": r"QVCache ($D=0.05$)",
                "aker": "Aker",
            },
            "compact_legend": True,
            "hatch_recall": False,
            "legend_ncol": 3,
            "latency": True,
            "latency_detail": False,
            "fig_w": 7.6,
            "panel_start": 8,
            "output": "spacev-10m-search-dev0.05-row3.pdf",
        },
        {
            "name": "sphere-10m-search",
            "kind": "search",
            "root": root / "logs/search/sphere-10m",
            "rows": sphere_search_rows,
            "cols": list(skews),
            "latency": True,
            "fig_w": 7.6,
            "output": "sphere-10m-search.pdf",
        },
        {
            "name": "sphere-10m-search-dev0.25",
            "kind": "search",
            "root": root / "logs/search/sphere-10m",
            "rows": sphere_search_rows,
            "cols": list(skews),
            "systems": ("qvcache", "qvcache-dev0.25", "aker"),
            "fig_w": 7.6,
            "output": "sphere-10m-search-dev0.25.pdf",
        },
        {
            "name": "sphere-10m-search-dev0.05",
            "kind": "search",
            "root": root / "logs/search/sphere-10m",
            "rows": sphere_search_rows,
            "cols": list(skews),
            "systems": ("qvcache", "qvcache-dev0.05", "aker"),
            "system_names": {
                "qvcache": r"QVCache (default, $D=0$)",
                "qvcache-dev0.05": r"QVCache ($D=0.05$)",
                "aker": "Aker",
            },
            "compact_legend": True,
            "hatch_recall": False,
            "legend_ncol": 4,
            "latency": True,
            "latency_detail": False,
            "fig_w": 7.6,
            "output": "sphere-10m-search-dev0.05.pdf",
        },
        {
            "name": "sphere-10m-sim-100k-0.99",
            "kind": "search",
            "root": root / "logs/search/sphere-10m",
            "rows": [({"0.99": "sim-100k-0.99"}, "simZipf")],
            "cols": [("0.99", r"$\alpha$ = 0.99")],
            "systems": (
                "qvcache",
                "qvcache-b4",
                "qvcache-dev0.01",
                "qvcache-dev0.05",
                "qvcache-dev0.25",
                "aker",
            ),
            "fig_w": 7.6,
            "fig_h": 4.4,
            "ylim": 148,
            "output": "sphere-10m-sim-100k-0.99.pdf",
        },
    ]
    full = next(spec for spec in specs if spec["name"] == "spacev-10m-search-dev0.05")
    peak = search_latency_peak(full)
    for spec in specs:
        if spec["name"].startswith("spacev-10m-search-dev0.05-row"):
            spec["latency_at_least"] = peak
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "figures",
        help="Directory for the PDF figures (default: figures/)",
    )
    args = parser.parse_args()
    apply_style()
    root = repo_root()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    for spec in figures(root):
        output = out_dir / spec["output"]
        render(spec, output)
        print(output)


if __name__ == "__main__":
    main()
