#!/usr/bin/env python3
"""Fixed k=10 vs k sampled from {1, 5, 10} on simZipf skew 0.99."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FIXED = "#2b6cb0"
VARY = "#dd6b20"
EDGE = "#1a202c"
METRICS = (
    ("hit", "Hit ratio", ".2f"),
    ("recall", "Recall", ".3f"),
)


def _json_payload(line: str) -> str | None:
    line = line.strip()
    if line.startswith("{"):
        return line
    start = line.find('{"event"')
    if start < 0:
        return None
    return line[start:]


def load_stream(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            payload = _json_payload(line)
            if payload is None or '"stream_metrics"' not in payload:
                continue
            rows.append(json.loads(payload))
    if not rows:
        raise SystemExit(f"No stream_metrics in {path}")
    return rows


def weighted(rows: list[dict], key: str, count_key: str) -> float:
    total = 0.0
    weight = 0.0
    for row in rows:
        count = row.get(count_key) or 0
        value = row.get(key)
        if count and value is not None:
            total += float(value) * float(count)
            weight += float(count)
    return total / weight if weight else np.nan


def summarize(rows: list[dict]) -> dict[str, float]:
    last = rows[-1]
    return {
        "hit": float(last["cum_hit_ratio"]),
        "qps": float(last["cum_qps"]),
        "latency_ms": float(last["cum_avg_latency_ms"]),
        "recall": weighted(rows, "recall_all", "interval_queries"),
        "hit_recall": weighted(rows, "recall_cache_hits", "cache_hit_count"),
        "hit_lat": weighted(rows, "avg_hit_latency_ms", "hits"),
    }


def style_ax(ax) -> None:
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(pairs: list[tuple[str, Path, Path]], output: Path) -> None:
    stats = []
    for label, fixed_path, vary_path in pairs:
        stats.append(
            (
                label,
                summarize(load_stream(fixed_path)),
                summarize(load_stream(vary_path)),
            )
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), sharey=True)
    x = np.arange(len(METRICS))
    x_lat = float(len(METRICS)) + 0.15
    width = 0.34
    ylim = 1.22
    for ax, (label, fixed, vary) in zip(axes, stats):
        fixed_vals = [fixed[key] for key, _, _ in METRICS]
        vary_vals = [vary[key] for key, _, _ in METRICS]
        ax.bar(
            x - width / 2,
            fixed_vals,
            width,
            color=FIXED,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            vary_vals,
            width,
            color=VARY,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        lat_ax = ax.twinx()
        lat_ax.patch.set_visible(False)
        lat_ax.bar(
            [x_lat - width / 2],
            [fixed["hit_lat"]],
            width,
            color=FIXED,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        lat_ax.bar(
            [x_lat + width / 2],
            [vary["hit_lat"]],
            width,
            color=VARY,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xticks([*x, x_lat])
        ax.set_xticklabels([name for _, name, _ in METRICS] + ["Hit latency"])
        ax.set_xlim(-0.55, x_lat + 0.55)
        ax.set_title(label)
        ax.set_ylim(0, ylim)
        lat_ax.set_ylim(0, ylim)
        style_ax(ax)
        lat_ax.spines["top"].set_visible(False)
        lat_ax.spines["left"].set_visible(False)
        lat_ax.tick_params(axis="y", labelsize=8)
        lat_ax.set_ylabel("Hit latency (ms)")
        for xpos, value, (_, _, fmt) in zip(x - width / 2, fixed_vals, METRICS):
            ax.text(xpos, value + 0.025, format(value, fmt), ha="center", va="bottom", fontsize=8)
        for xpos, value, (_, _, fmt) in zip(x + width / 2, vary_vals, METRICS):
            ax.text(xpos, value + 0.025, format(value, fmt), ha="center", va="bottom", fontsize=8)
        lat_ax.text(
            x_lat - width / 2,
            fixed["hit_lat"] + ylim * 0.02,
            f"{fixed['hit_lat']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        lat_ax.text(
            x_lat + width / 2,
            vary["hit_lat"] + ylim * 0.02,
            f"{vary['hit_lat']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_ylabel("Ratio")

    legend = [
        Patch(facecolor=FIXED, edgecolor=EDGE, label=r"Fixed $k = 10$"),
        Patch(facecolor=VARY, edgecolor=EDGE, label=r"Vary $k \in \{1, 5, 10\}$"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    if output.suffix.lower() != ".pdf":
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Wrote {pdf}")
    print(f"Wrote {output}")
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=repo / "logs" / "fixed_vs_vary_k.png")
    parser.add_argument("--fixed-simzipf", type=Path, default=repo / "fixedk_simzipf99.log")
    parser.add_argument("--vary-simzipf", type=Path, default=repo / "varyk_simzipf99.log")
    parser.add_argument("--fixed-simzipf-plus", type=Path, default=repo / "fixedk_simzipfplus99.log")
    parser.add_argument("--vary-simzipf-plus", type=Path, default=repo / "varyk_simzipfplus99.log")
    parser.add_argument("--fixed-simzipf-plus2", type=Path, default=repo / "fixedk_simzipfplus299.log")
    parser.add_argument("--vary-simzipf-plus2", type=Path, default=repo / "varyk_simzipfplus299.log")
    args = parser.parse_args()
    plot(
        [
            (r"simZipf ($\lambda \sim U(0, 0.5)$)", args.fixed_simzipf, args.vary_simzipf),
            (r"simZipf+ ($\lambda = 0.01$)", args.fixed_simzipf_plus, args.vary_simzipf_plus),
            (r"simZipf+ ($\lambda \sim U(0, 0.5)$)", args.fixed_simzipf_plus2, args.vary_simzipf_plus2),
        ],
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
