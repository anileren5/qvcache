#!/usr/bin/env python3
"""Paper-quality 3x4 bar charts of hit ratio and cache-hit recall.

Rows are simZipf and two simZipf+ ε settings. Columns are uniform, 0.3, 0.6, 0.99.
QVCache hits are all approximate. Aker hit-ratio bars are stacked exact/approx.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_sim_workload_logs import parse_metrics, summarize  # noqa: E402

WORKLOADS = (
    ("simZipf", r"simZipf ($\lambda \sim U(0, 0.5)$)"),
    ("simZipf+", r"simZipf+ ($\lambda = 0.01$)"),
    ("simZipf+2", r"simZipf+ ($\lambda \sim U(0, 0.5)$)"),
)
PARAMS = (
    ("uniform", "Uniform"),
    ("0.3", r"Zipf $0.30$"),
    ("0.6", r"Zipf $0.60$"),
    ("0.99", r"Zipf $0.99$"),
)

QV_COLOR = "#2b6cb0"
AKER_EXACT = "#9b2c2c"
AKER_APPROX = "#dd6b20"
EDGE = "#1a202c"


def _weighted_avg(records: list[dict], key: str, *, count_key: str | None = None) -> float:
    weighted = 0.0
    weight = 0.0
    for rec in records:
        if count_key is None:
            n = rec.get("interval_queries", rec.get("total_queries", 0)) or 0
        elif count_key == "misses":
            nq = rec.get("interval_queries", rec.get("total_queries", 0)) or 0
            hits = rec.get("hits", 0) or 0
            n = max(float(nq) - float(hits), 0.0)
        else:
            n = rec.get(count_key, 0) or 0
        val = rec.get(key)
        if n and val is not None:
            weighted += float(val) * float(n)
            weight += float(n)
    return weighted / weight if weight else np.nan


def load_cell(log_dir: Path, workload: str, param: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for system in ("qvcache", "aker"):
        path = log_dir / workload / f"{system}_{param}.log"
        if not path.is_file():
            raise SystemExit(f"Missing log: {path}")
        records = parse_metrics(path)
        stats = summarize(records)
        hit = float(stats["avg_cache_hits"])
        exact = float(stats["avg_exact_hit_ratio"])
        approx = float(stats["avg_approx_hit_ratio"])
        recall = float(stats["avg_recall_cache_hits"])
        hit_lat = float(stats["avg_hit_latency"])
        miss_pen = float(_weighted_avg(records, "miss_penalty_ms", count_key="misses"))
        avg_lat = float(_weighted_avg(records, "avg_latency_ms"))
        if system == "qvcache" or not np.isfinite(exact):
            exact = 0.0
            approx = hit if np.isfinite(hit) else 0.0
        elif np.isfinite(exact) and np.isfinite(approx):
            stacked = exact + approx
            if not np.isfinite(hit):
                hit = stacked
        out[system] = {
            "hit": hit if np.isfinite(hit) else 0.0,
            "exact": exact if np.isfinite(exact) else 0.0,
            "approx": approx if np.isfinite(approx) else 0.0,
            "recall": recall if np.isfinite(recall) else np.nan,
            "hit_lat": hit_lat if np.isfinite(hit_lat) else np.nan,
            "miss_pen": miss_pen if np.isfinite(miss_pen) else np.nan,
            "avg_lat": avg_lat if np.isfinite(avg_lat) else np.nan,
        }
    qv = out["qvcache"]
    qv["exact"] = 0.0
    qv["approx"] = qv["hit"]
    return out


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) < 0.005:
        return ".00"
    return f"{value:.2f}".lstrip("0") if value < 1 else f"{value:.2f}"


def _fmt_ms(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.01:
        return f"{value:.3f}"
    if value < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def annotate(ax, x: float, y: float, text: str, *, va: str, color: str = "black") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va=va,
        fontsize=7.2,
        color=color,
        clip_on=False,
    )


def draw_hit_bar(ax, x: float, width: float, exact: float, approx: float, *, qv: bool) -> float:
    total = exact + approx
    if qv:
        ax.bar(
            x,
            total,
            width=width,
            color=QV_COLOR,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        if total >= 0.045:
            annotate(ax, x, total / 2.0, _fmt(total), va="center", color="white")
        else:
            annotate(ax, x, total + 0.012, _fmt(total), va="bottom")
        return total

    if exact > 0:
        ax.bar(
            x,
            exact,
            width=width,
            color=AKER_EXACT,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        if exact >= 0.05:
            annotate(ax, x, exact / 2.0, _fmt(exact), va="center", color="white")
    if approx > 0:
        ax.bar(
            x,
            approx,
            width=width,
            bottom=exact,
            color=AKER_APPROX,
            edgecolor=EDGE,
            linewidth=0.6,
            zorder=3,
        )
        if approx >= 0.05:
            annotate(
                ax,
                x,
                exact + approx / 2.0,
                _fmt(approx),
                va="center",
                color="white",
            )
    if total > 0:
        # Total on top when the stack is split, so the bar still reads as a ratio.
        if exact > 0 and approx > 0:
            annotate(ax, x, total + 0.012, _fmt(total), va="bottom")
        elif exact < 0.05 and approx < 0.05:
            annotate(ax, x, total + 0.012, _fmt(total), va="bottom")
        elif exact > 0 and approx == 0 and exact < 0.05:
            annotate(ax, x, total + 0.012, _fmt(exact), va="bottom")
        elif approx > 0 and exact == 0 and approx < 0.05:
            annotate(ax, x, total + 0.012, _fmt(approx), va="bottom")
    return total


def draw_recall_bar(ax, x: float, width: float, recall: float, *, qv: bool) -> None:
    if not np.isfinite(recall):
        annotate(ax, x, 0.012, "—", va="bottom")
        return
    ax.bar(
        x,
        recall,
        width=width,
        color=QV_COLOR if qv else AKER_APPROX,
        edgecolor=EDGE,
        linewidth=0.6,
        zorder=3,
    )
    if recall >= 0.08:
        annotate(ax, x, recall / 2.0, _fmt(recall), va="center", color="white")
    else:
        annotate(ax, x, recall + 0.012, _fmt(recall), va="bottom")


def plot(log_dir: Path, output: Path) -> None:
    cells = {
        (wl, param): load_cell(log_dir, wl, param)
        for wl, _ in WORKLOADS
        for param, _ in PARAMS
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )

    fig, axes = plt.subplots(
        len(WORKLOADS),
        len(PARAMS),
        figsize=(12.4, 8.8),
        sharey=True,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.07, wspace=0.18, hspace=0.32)

    width = 0.38
    x_hit = np.array([-0.22, 0.22])
    x_rec = np.array([1.28, 1.72])

    for i, (wl, wl_label) in enumerate(WORKLOADS):
        for j, (param, param_label) in enumerate(PARAMS):
            ax = axes[i, j]
            stats = cells[(wl, param)]
            qv, aker = stats["qvcache"], stats["aker"]

            draw_hit_bar(ax, x_hit[0], width, qv["exact"], qv["approx"], qv=True)
            draw_hit_bar(ax, x_hit[1], width, aker["exact"], aker["approx"], qv=False)
            draw_recall_bar(ax, x_rec[0], width, qv["recall"], qv=True)
            draw_recall_bar(ax, x_rec[1], width, aker["recall"], qv=False)

            ax.set_xlim(-0.7, 2.2)
            ax.set_ylim(0.0, 1.08)
            ax.set_xticks([0.0, 1.5])
            ax.set_xticklabels(["Hit ratio", "Recall"], fontsize=8.5)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75", zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                ax.set_title(param_label, pad=6)
            if j == 0:
                ax.set_ylabel(wl_label, fontsize=11)
            if j != 0:
                ax.tick_params(axis="y", labelleft=False)
            ax.text(
                0.03,
                0.97,
                (
                    "QV / Aker\n"
                    f"avg   {_fmt_ms(qv['avg_lat'])} / {_fmt_ms(aker['avg_lat'])} ms\n"
                    f"hit   {_fmt_ms(qv['hit_lat'])} / {_fmt_ms(aker['hit_lat'])} ms\n"
                    f"miss  {_fmt_ms(qv['miss_pen'])} / {_fmt_ms(aker['miss_pen'])} ms"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color="0.1",
                linespacing=1.35,
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "facecolor": "white",
                    "edgecolor": "#718096",
                    "linewidth": 0.7,
                    "alpha": 0.95,
                },
            )

    legend = [
        Patch(facecolor=QV_COLOR, edgecolor=EDGE, label="QVCache (approx.)"),
        Patch(facecolor=AKER_EXACT, edgecolor=EDGE, label="Aker exact"),
        Patch(facecolor=AKER_APPROX, edgecolor=EDGE, label="Aker approx."),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    if output.suffix.lower() != ".pdf":
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Wrote {pdf}")
    print(f"Wrote {output}")

    print("workload\tparam\tsystem\thit\texact\tapprox\trecall\tavg_ms\thit_ms\tmiss_ms")
    for wl, _ in WORKLOADS:
        for param, _ in PARAMS:
            for system in ("qvcache", "aker"):
                s = cells[(wl, param)][system]
                rec = s["recall"]
                rec_s = f"{rec:.4f}" if np.isfinite(rec) else "nan"
                print(
                    f"{wl}\t{param}\t{system}\t{s['hit']:.4f}\t"
                    f"{s['exact']:.4f}\t{s['approx']:.4f}\t{rec_s}\t"
                    f"{_fmt_ms(s['avg_lat'])}\t{_fmt_ms(s['hit_lat'])}\t{_fmt_ms(s['miss_pen'])}"
                )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=repo / "logs",
        help="Root log directory containing simZipf, simZipf+, simZipf+2",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo / "logs" / "sim_hit_recall_bars.png",
        help="Output image path (PDF is also written alongside PNG)",
    )
    args = parser.parse_args()
    plot(args.logs_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
