#!/usr/bin/env python3
"""Plot stream_metrics or window_metrics from two search-workload logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_METRICS = [
    {
        "key": "hit_ratio",
        "title": "Hit ratio",
        "ylabel": "Hit ratio",
        "mask_zero_hits": False,
    },
    {
        "key": "exact_hit_ratio",
        "title": "Exact hit ratio",
        "ylabel": "Hit ratio",
        "mask_zero_hits": False,
        "optional": True,
    },
    {
        "key": "approx_hit_ratio",
        "title": "Approx hit ratio",
        "ylabel": "Hit ratio",
        "mask_zero_hits": False,
        "optional": True,
    },
    {
        "key": "avg_hit_latency_ms",
        "title": "Avg hit latency (all)",
        "ylabel": "ms",
        "mask_zero_hits": True,
    },
    {
        "key": "avg_exact_hit_latency_ms",
        "title": "Avg exact hit latency",
        "ylabel": "ms",
        "mask_zero_hits": True,
        "mask_count_key": "exact_hits",
    },
    {
        "key": "avg_approx_hit_latency_ms",
        "title": "Avg approx hit latency",
        "ylabel": "ms",
        "mask_zero_hits": True,
        "mask_count_key": "approx_hits",
    },
    {
        "key": "miss_penalty_ms",
        "title": "Miss penalty",
        "ylabel": "ms",
        "mask_zero_hits": False,
    },
    {
        "key": "recall_cache_hits",
        "title": "Recall (cache hits)",
        "ylabel": "Recall@K",
        "mask_zero_hits": True,
    },
    {
        "key": "hits",
        "title": "Cache hits",
        "ylabel": "Hits / report",
        "mask_zero_hits": False,
    },
]


AKER_HIT_BREAKDOWN_KEYS = (
    "exact_hits",
    "approx_hits",
    "exact_hit_ratio",
    "approx_hit_ratio",
    "avg_exact_hit_latency_ms",
    "avg_approx_hit_latency_ms",
)


def is_qvcache_series(path: Path, records: list[dict]) -> bool:
    cache = records[0].get("cache") if records else None
    if cache == "qvcache":
        return True
    if cache == "aker":
        return False
    return "qvcache" in path.stem.lower()


def drop_aker_hit_breakdown(records: list[dict]) -> None:
    for rec in records:
        for key in AKER_HIT_BREAKDOWN_KEYS:
            rec.pop(key, None)


METRIC_EVENTS = {"stream_metrics", "window_metrics"}


def _json_payload(line: str) -> str | None:
    line = line.strip()
    if line.startswith("{"):
        return line
    start = line.find('{"event"')
    if start < 0:
        return None
    return line[start:]


def parse_metrics(path: Path) -> list[dict]:
    records = []
    completed = 0
    with path.open() as f:
        for line in f:
            payload = _json_payload(line)
            if payload is None or '"event"' not in payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("event") not in METRIC_EVENTS:
                continue
            queries = obj.get("interval_queries", obj.get("total_queries", 0)) or 0
            obj["interval_queries"] = queries
            if "completed" not in obj:
                completed += queries
                obj["completed"] = completed
            records.append(obj)
    if not records:
        raise SystemExit(
            f"No stream_metrics or window_metrics records in {path}"
        )
    return records


def series(
    records: list[dict],
    key: str,
    mask_zero_hits: bool,
    mask_count_key: str | None = None,
) -> np.ndarray:
    values = []
    for rec in records:
        raw = rec.get(key)
        if raw is None:
            values.append(np.nan)
            continue
        if mask_zero_hits:
            if mask_count_key is not None:
                count = rec.get(mask_count_key, 0) or 0
                if count == 0:
                    values.append(np.nan)
                    continue
            else:
                hits = rec.get("hits", rec.get("cache_hit_count", 0)) or 0
                hit_ratio = rec.get("hit_ratio", 0) or 0
                if hits == 0 or hit_ratio == 0:
                    values.append(np.nan)
                    continue
        values.append(float(raw))
    return np.asarray(values, dtype=float)


def has_metric(records: list[dict], key: str) -> bool:
    return any(rec.get(key) is not None for rec in records)


def select_metrics(left: list[dict], right: list[dict]) -> list[dict]:
    selected = []
    for spec in BASE_METRICS:
        if spec.get("optional") and not (
            has_metric(left, spec["key"]) or has_metric(right, spec["key"])
        ):
            continue
        selected.append(spec)
    return selected


def label_for(path: Path, records: list[dict]) -> str:
    cache = records[0].get("cache")
    if cache:
        return f"{cache} ({path.name})"
    name = path.stem.lower()
    if "qvcache" in name:
        return f"qvcache ({path.name})"
    if "aker" in name:
        return f"aker ({path.name})"
    return path.name


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan)
    p50, p90, p99 = np.percentile(finite, [50, 90, 99])
    return float(p50), float(p90), float(p99)


def summarize(records: list[dict]) -> dict[str, float]:
    hits = series(records, "hits", False)
    interval_queries = np.asarray(
        [rec.get("interval_queries", rec.get("total_queries", 0)) or 0 for rec in records],
        dtype=float,
    )
    total_hits = float(np.nansum(hits))
    total_queries = float(np.nansum(interval_queries))
    avg_cache_hits = total_hits / total_queries if total_queries else np.nan

    def _count_ratio(count_key: str, ratio_key: str) -> float:
        if not has_metric(records, count_key) and not has_metric(records, ratio_key):
            return np.nan
        counted = 0.0
        queried = 0.0
        for rec, nq in zip(records, interval_queries):
            if nq <= 0:
                continue
            if rec.get(count_key) is not None:
                counted += float(rec[count_key])
                queried += nq
            elif rec.get(ratio_key) is not None:
                counted += float(rec[ratio_key]) * nq
                queried += nq
        return counted / queried if queried else np.nan

    avg_exact_hit_ratio = _count_ratio("exact_hits", "exact_hit_ratio")
    avg_approx_hit_ratio = _count_ratio("approx_hits", "approx_hit_ratio")

    weighted_recall = 0.0
    recall_weight = 0.0
    for rec in records:
        count = rec.get("hits", rec.get("cache_hit_count", 0)) or 0
        recall = rec.get("recall_cache_hits")
        if count and recall is not None:
            weighted_recall += float(recall) * float(count)
            recall_weight += float(count)
    avg_recall_cache_hits = (
        weighted_recall / recall_weight if recall_weight else np.nan
    )

    miss_p50, miss_p90, miss_p99 = _percentiles(
        series(records, "miss_penalty_ms", False)
    )
    hit_p50, hit_p90, hit_p99 = _percentiles(
        series(records, "avg_hit_latency_ms", True)
    )
    exact_lat_p50, exact_lat_p90, exact_lat_p99 = _percentiles(
        series(records, "avg_exact_hit_latency_ms", True, "exact_hits")
    )
    approx_lat_p50, approx_lat_p90, approx_lat_p99 = _percentiles(
        series(records, "avg_approx_hit_latency_ms", True, "approx_hits")
    )

    def _weighted_latency(latency_key: str, count_key: str) -> float:
        if not has_metric(records, latency_key):
            return np.nan
        weighted = 0.0
        weight = 0.0
        for rec in records:
            count = rec.get(count_key, 0) or 0
            latency = rec.get(latency_key)
            if count and latency is not None:
                weighted += float(latency) * float(count)
                weight += float(count)
        return weighted / weight if weight else np.nan

    avg_hit_latency = _weighted_latency("avg_hit_latency_ms", "hits")
    avg_exact_hit_latency = _weighted_latency(
        "avg_exact_hit_latency_ms", "exact_hits"
    )
    avg_approx_hit_latency = _weighted_latency(
        "avg_approx_hit_latency_ms", "approx_hits"
    )
    return {
        "avg_cache_hits": avg_cache_hits,
        "avg_exact_hit_ratio": avg_exact_hit_ratio,
        "avg_approx_hit_ratio": avg_approx_hit_ratio,
        "avg_recall_cache_hits": avg_recall_cache_hits,
        "miss_p50": miss_p50,
        "miss_p90": miss_p90,
        "miss_p99": miss_p99,
        "avg_hit_latency": avg_hit_latency,
        "hit_p50": hit_p50,
        "hit_p90": hit_p90,
        "hit_p99": hit_p99,
        "avg_exact_hit_latency": avg_exact_hit_latency,
        "exact_lat_p50": exact_lat_p50,
        "exact_lat_p90": exact_lat_p90,
        "exact_lat_p99": exact_lat_p99,
        "avg_approx_hit_latency": avg_approx_hit_latency,
        "approx_lat_p50": approx_lat_p50,
        "approx_lat_p90": approx_lat_p90,
        "approx_lat_p99": approx_lat_p99,
    }


def short_label(label: str) -> str:
    return label.split(" (", 1)[0]


def summary_pairs(
    stats: dict[str, float],
    *,
    breakdown: bool,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Hit ratio", _fmt(stats["avg_cache_hits"], 4)),
    ]
    if breakdown:
        rows.append(("Approx hit ratio", _fmt(stats["avg_approx_hit_ratio"], 4)))
        rows.append(("Exact hit ratio", _fmt(stats["avg_exact_hit_ratio"], 4)))
    rows.extend(
        [
            ("Avg recall cache hits", _fmt(stats["avg_recall_cache_hits"], 4)),
            ("Miss p50 (ms)", _fmt(stats["miss_p50"])),
            ("Miss p90 (ms)", _fmt(stats["miss_p90"])),
            ("Miss p99 (ms)", _fmt(stats["miss_p99"])),
            ("Avg hit lat (ms)", _fmt(stats["avg_hit_latency"])),
            ("Hit lat p50 (ms)", _fmt(stats["hit_p50"])),
            ("Hit lat p90 (ms)", _fmt(stats["hit_p90"])),
            ("Hit lat p99 (ms)", _fmt(stats["hit_p99"])),
            ("Avg exact hit lat (ms)", _fmt(stats["avg_exact_hit_latency"])),
            ("Exact hit lat p50 (ms)", _fmt(stats["exact_lat_p50"])),
            ("Exact hit lat p90 (ms)", _fmt(stats["exact_lat_p90"])),
            ("Exact hit lat p99 (ms)", _fmt(stats["exact_lat_p99"])),
            ("Avg approx hit lat (ms)", _fmt(stats["avg_approx_hit_latency"])),
            ("Approx hit lat p50 (ms)", _fmt(stats["approx_lat_p50"])),
            ("Approx hit lat p90 (ms)", _fmt(stats["approx_lat_p90"])),
            ("Approx hit lat p99 (ms)", _fmt(stats["approx_lat_p99"])),
        ]
    )
    return rows


def _x_axis(records: list[dict]) -> tuple[np.ndarray, str, list[str] | None]:
    windowed = all(rec.get("event") == "window_metrics" for rec in records)
    if windowed:
        x = np.arange(len(records), dtype=float)
        labels = [
            f"{rec.get('window_idx', i)}.{rec.get('repeat_idx', 0)}"
            for i, rec in enumerate(records)
        ]
        return x, "Window.repeat", labels
    x = np.asarray([rec["completed"] for rec in records], dtype=float)
    return x, "Completed queries", None


def plot_logs(left: Path, right: Path, output: Path) -> None:
    a = parse_metrics(left)
    b = parse_metrics(right)
    if is_qvcache_series(left, a):
        drop_aker_hit_breakdown(a)
    if is_qvcache_series(right, b):
        drop_aker_hit_breakdown(b)
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(
            f"Aligned to {n} reports "
            f"({left.name}: {len(a)}, {right.name}: {len(b)})"
        )
    a = a[:n]
    b = b[:n]

    x, xlabel, xticklabels = _x_axis(a)
    label_a = label_for(left, a)
    label_b = label_for(right, b)
    metrics = []
    for spec in select_metrics(a, b):
        ya = series(
            a, spec["key"], spec["mask_zero_hits"], spec.get("mask_count_key")
        )
        yb = series(
            b, spec["key"], spec["mask_zero_hits"], spec.get("mask_count_key")
        )
        if spec["key"] in {
            "exact_hit_ratio",
            "approx_hit_ratio",
            "avg_exact_hit_latency_ms",
            "avg_approx_hit_latency_ms",
        } and not (np.isfinite(ya).any() or np.isfinite(yb).any()):
            continue
        metrics.append(spec)
    show_breakdown = any(
        spec["key"] in {"exact_hit_ratio", "approx_hit_ratio"} for spec in metrics
    )

    fig, axes = plt.subplots(
        len(metrics) + 1,
        1,
        figsize=(12.5, 2.05 * len(metrics) + 14.5),
        sharex=False,
        gridspec_kw={"height_ratios": [1] * len(metrics) + [4.4]},
    )
    plot_axes = axes[:-1]
    table_ax = axes[-1]
    fig.suptitle("Search-workload metrics", fontsize=14)

    for ax, spec in zip(plot_axes, metrics):
        ya = series(
            a, spec["key"], spec["mask_zero_hits"], spec.get("mask_count_key")
        )
        yb = series(
            b, spec["key"], spec["mask_zero_hits"], spec.get("mask_count_key")
        )
        ax.plot(x, ya, label=label_a, linewidth=1.6)
        ax.plot(x, yb, label=label_b, linewidth=1.6)
        ax.set_title(spec["title"])
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        if spec["mask_zero_hits"]:
            ax.set_ylim(bottom=0)
            ax.text(
                0.01,
                0.97,
                "Gaps: no hits of this type in that interval (value is undefined, not 0)",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="0.35",
            )

    plot_axes[-1].set_xlabel(xlabel)
    for i, ax in enumerate(plot_axes):
        if len(x) == 1:
            ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
        else:
            ax.set_xlim(x.min(), x.max())
        if xticklabels is not None:
            ax.set_xticks(x)
            if i == len(plot_axes) - 1:
                ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])

    stats_a = summarize(a)
    stats_b = summarize(b)
    pairs_a = summary_pairs(stats_a, breakdown=show_breakdown)
    pairs_b = summary_pairs(stats_b, breakdown=show_breakdown)
    headers = ["Metric", short_label(label_a), short_label(label_b)]
    cell_text = [
        [name, value_a, value_b]
        for (name, value_a), (_, value_b) in zip(pairs_a, pairs_b)
    ]
    table_ax.axis("off")
    n_body = len(cell_text)
    table_ax.set_title(
        "Table 1. Workload summary (aligned reports).",
        fontsize=15,
        pad=18,
        fontfamily="serif",
        fontstyle="italic",
    )
    table = table_ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.50, 0.25, 0.25],
        bbox=[0.05, 0.03, 0.90, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.05, 1.7)

    header_fill = "#f3f3f3"
    stripe = "#fbfbfb"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.PAD = 0.12
        text = cell.get_text()
        text.set_fontfamily("serif")
        text.set_fontsize(13)
        if row == 0:
            cell.set_facecolor(header_fill)
            cell.visible_edges = "TB"
            cell.set_linewidth(1.8)
            text.set_weight("bold")
            text.set_ha("left" if col == 0 else "center")
            continue
        cell.set_facecolor(stripe if row % 2 == 0 else "white")
        if row == n_body:
            cell.visible_edges = "B"
            cell.set_linewidth(1.8)
        else:
            cell.visible_edges = ""
            cell.set_linewidth(0)
        text.set_ha("left" if col == 0 else "right")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")
    print("\t".join(headers))
    for row in cell_text:
        print("\t".join(row))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_a", type=Path, help="First log file")
    parser.add_argument("log_b", type=Path, help="Second log file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo / "qvcache_vs_aker.png",
        help="Output image path",
    )
    args = parser.parse_args()
    plot_logs(args.log_a, args.log_b, args.output)


if __name__ == "__main__":
    main()
