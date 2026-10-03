#!/usr/bin/env python3
"""Paper-quality hit-ratio and cache-hit recall bars for refresh workloads.

Rows are simZipf and simZipf+. Columns are delete rates. Each cell is the
post-refresh eval pass (5% insert, then the column's delete rate). QVCache
hits are all approximate. Refresh logs do not split Aker hits into exact and
approximate, so Aker is a single bar in the same orange used for Aker recall
in the search-only figure. The inset is mean per-operation insert and delete
latency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_sim_workload_logs import _json_payload  # noqa: E402

WORKLOADS = (
    ("simZipf", r"simZipf ($\lambda \sim U(0, 0.5)$)"),
    ("simZipf+", r"simZipf+ ($\lambda = 0.01$)"),
)
RATES = (
    (0.05, r"Delete $5\%$" "\n" r"Insert $5\%$"),
    (0.10, r"Delete $10\%$" "\n" r"Insert $5\%$"),
    (0.25, r"Delete $25\%$" "\n" r"Insert $5\%$"),
)

QV_COLOR = "#2b6cb0"
AKER_COLOR = "#dd6b20"
EDGE = "#1a202c"


def _rate_key(rate: float) -> str:
    return f"{float(rate):.2f}"


def _avg_ms(event: dict | None) -> float:
    if not event or event.get("avg_ms") is None:
        return np.nan
    value = float(event["avg_ms"])
    return value if np.isfinite(value) else np.nan


def load_eval(path: Path) -> dict:
    start = None
    end = None
    insert = None
    delete = None
    with path.open() as f:
        for line in f:
            payload = _json_payload(line)
            if payload is None or '"event"' not in payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            event = obj.get("event")
            if event == "refresh_start":
                start = obj
            elif event == "refresh_end":
                end = obj
            elif event == "refresh_insert":
                insert = obj
            elif event == "refresh_delete":
                delete = obj
    if start is None or end is None:
        raise SystemExit(f"Missing refresh_start/refresh_end in {path}")
    cache = str(start.get("cache") or end.get("cache") or "")
    if cache not in ("qvcache", "aker"):
        raise SystemExit(f"Unknown cache {cache!r} in {path}")
    hit = float(end["eval_hit_ratio"])
    recall = float(end["eval_recall_hits"])
    return {
        "cache": cache,
        "delete_rate": float(start["delete_rate"]),
        "insert_frac": float(start.get("insert_frac", np.nan)),
        "hit": hit if np.isfinite(hit) else np.nan,
        "recall": recall if np.isfinite(recall) else np.nan,
        "insert_ms": _avg_ms(insert),
        "delete_ms": _avg_ms(delete),
        "path": path,
    }


def load_cells(dirs: dict[str, Path]) -> dict[tuple[str, str], dict[str, dict]]:
    cells: dict[tuple[str, str], dict[str, dict]] = {}
    expected_rates = {_rate_key(rate) for rate, _ in RATES}
    for workload, directory in dirs.items():
        if not directory.is_dir():
            raise SystemExit(f"Missing log directory: {directory}")
        found: dict[tuple[str, str], dict] = {}
        logs = sorted(directory.rglob("*.log"))
        if not logs:
            raise SystemExit(f"No logs in {directory}")
        for path in logs:
            rec = load_eval(path)
            key = (_rate_key(rec["delete_rate"]), rec["cache"])
            if key in found:
                raise SystemExit(
                    f"Duplicate {rec['cache']} delete_rate={rec['delete_rate']} "
                    f"in {directory}: {found[key]['path']} and {path}"
                )
            found[key] = rec
        for rate, _ in RATES:
            rate_s = _rate_key(rate)
            stats = {}
            for system in ("qvcache", "aker"):
                rec = found.get((rate_s, system))
                if rec is None:
                    raise SystemExit(
                        f"Missing {system} delete_rate={rate_s} under {directory}"
                    )
                stats[system] = rec
            inserts = {stats[s]["insert_frac"] for s in ("qvcache", "aker")}
            if len(inserts) != 1:
                raise SystemExit(
                    f"Insert fraction mismatch for {workload} delete_rate={rate_s}: {inserts}"
                )
            cells[(workload, rate_s)] = stats
        extra = {rate for rate, _ in found} - expected_rates
        if extra:
            print(f"Ignoring extra delete rates in {directory}: {sorted(extra)}", file=sys.stderr)
    return cells


def _fmt_ms(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.01:
        return f"{value:.3f}"
    if value < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) < 0.005:
        return ".00"
    return f"{value:.2f}".lstrip("0") if value < 1 else f"{value:.2f}"


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


def draw_bar(ax, x: float, width: float, value: float, *, color: str, inside: float) -> None:
    if not np.isfinite(value):
        annotate(ax, x, 0.012, "—", va="bottom")
        return
    ax.bar(
        x,
        value,
        width=width,
        color=color,
        edgecolor=EDGE,
        linewidth=0.6,
        zorder=3,
    )
    if value >= inside:
        annotate(ax, x, value / 2.0, _fmt(value), va="center", color="white")
    else:
        annotate(ax, x, value + 0.012, _fmt(value), va="bottom")


def plot(dirs: dict[str, Path], output: Path) -> None:
    cells = load_cells(dirs)

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
        len(RATES),
        figsize=(9.6, 6.6),
        sharey=True,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.07, wspace=0.18, hspace=0.55)

    width = 0.38
    x_hit = np.array([-0.22, 0.22])
    x_rec = np.array([1.28, 1.72])

    for i, (wl, wl_label) in enumerate(WORKLOADS):
        for j, (rate, rate_label) in enumerate(RATES):
            ax = axes[i, j]
            stats = cells[(wl, _rate_key(rate))]
            qv, aker = stats["qvcache"], stats["aker"]

            draw_bar(ax, x_hit[0], width, qv["hit"], color=QV_COLOR, inside=0.045)
            draw_bar(ax, x_hit[1], width, aker["hit"], color=AKER_COLOR, inside=0.045)
            draw_bar(ax, x_rec[0], width, qv["recall"], color=QV_COLOR, inside=0.08)
            draw_bar(ax, x_rec[1], width, aker["recall"], color=AKER_COLOR, inside=0.08)

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
                ax.set_title(rate_label, pad=6, linespacing=1.25)
            if j == 0:
                ax.set_ylabel(wl_label, fontsize=11)
            if j != 0:
                ax.tick_params(axis="y", labelleft=False)
            # Bottom-row bars reach the top of the axes, so those latencies
            # sit just above the frame. The top row still fits inside.
            ax.text(
                0.02,
                0.98 if i == 0 else 1.03,
                (
                    "QV / Aker\n"
                    f"ins   {_fmt_ms(qv['insert_ms'])} / {_fmt_ms(aker['insert_ms'])} ms\n"
                    f"del   {_fmt_ms(qv['delete_ms'])} / {_fmt_ms(aker['delete_ms'])} ms"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top" if i == 0 else "bottom",
                fontsize=7.0,
                color="0.1",
                linespacing=1.35,
                zorder=6,
                clip_on=False,
            )

    legend = [
        Patch(facecolor=QV_COLOR, edgecolor=EDGE, label="QVCache"),
        Patch(facecolor=AKER_COLOR, edgecolor=EDGE, label="Aker"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    if output.suffix.lower() != ".pdf":
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Wrote {pdf}")
    print(f"Wrote {output}")

    print("workload\tdelete_rate\tinsert_frac\tsystem\thit\trecall_hits\tinsert_ms\tdelete_ms")
    for wl, _ in WORKLOADS:
        for rate, _ in RATES:
            for system in ("qvcache", "aker"):
                s = cells[(wl, _rate_key(rate))][system]
                rec = s["recall"]
                rec_s = f"{rec:.4f}" if np.isfinite(rec) else "nan"
                hit_s = f"{s['hit']:.4f}" if np.isfinite(s["hit"]) else "nan"
                print(
                    f"{wl}\t{_rate_key(rate)}\t{s['insert_frac']:.2f}\t{system}\t"
                    f"{hit_s}\t{rec_s}\t{_fmt_ms(s['insert_ms'])}\t{_fmt_ms(s['delete_ms'])}"
                )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simzipf",
        type=Path,
        default=repo / "logs" / "simZipf_refresh",
        help="Directory of simZipf refresh logs",
    )
    parser.add_argument(
        "--simzipf-plus",
        type=Path,
        default=repo / "logs" / "simZipf+_refresh",
        help="Directory of simZipf+ refresh logs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo / "logs" / "sim_refresh_hit_recall_bars.png",
        help="Output image path (PDF is also written alongside PNG)",
    )
    args = parser.parse_args()
    plot(
        {"simZipf": args.simzipf, "simZipf+": args.simzipf_plus},
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
