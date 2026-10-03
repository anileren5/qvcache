#!/usr/bin/env python3
"""Hit ratio and cache-hit recall: 5% insert + 25% delete versus delete only.

Rows are simZipf and simZipf+. Columns are the insert-and-delete run and the
delete-only run. Same colors and bar layout as the refresh figure.
Delete-only logs have no insert phase, so that insert latency is omitted.
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

from plot_refresh_hit_recall_bars import (  # noqa: E402
    AKER_COLOR,
    EDGE,
    QV_COLOR,
    _fmt_ms,
    draw_bar,
    load_eval,
)

COLUMNS = (
    (
        "insert_delete",
        r"Delete $25\%$" "\n" r"Insert $5\%$",
    ),
    (
        "delete_only",
        r"Delete $25\%$" "\n" r"Insert $0\%$",
    ),
)
ROWS = (
    ("simZipf", r"simZipf ($\lambda \sim U(0, 0.5)$)"),
    ("simZipf+", r"simZipf+ ($\lambda = 0.01$)"),
)


def _pair(qv_path: Path, aker_path: Path) -> dict[str, dict]:
    qv = load_eval(qv_path)
    aker = load_eval(aker_path)
    if qv["cache"] != "qvcache" or aker["cache"] != "aker":
        raise SystemExit(f"Expected qvcache and aker logs, got {qv_path} and {aker_path}")
    return {"qvcache": qv, "aker": aker}


def _latency_text(qv: dict, aker: dict) -> str:
    lines = ["QV / Aker"]
    if np.isfinite(qv["insert_ms"]) or np.isfinite(aker["insert_ms"]):
        lines.append(
            f"ins   {_fmt_ms(qv['insert_ms'])} / {_fmt_ms(aker['insert_ms'])} ms"
        )
    lines.append(
        f"del   {_fmt_ms(qv['delete_ms'])} / {_fmt_ms(aker['delete_ms'])} ms"
    )
    return "\n".join(lines)


def plot(cells: dict[tuple[str, str], dict[str, dict]], output: Path) -> None:
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
        len(ROWS),
        len(COLUMNS),
        figsize=(6.8, 6.6),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.86, bottom=0.07, wspace=0.22, hspace=0.55)

    width = 0.38
    x_hit = np.array([-0.22, 0.22])
    x_rec = np.array([1.28, 1.72])

    for i, (workload, row_label) in enumerate(ROWS):
        for j, (key, title) in enumerate(COLUMNS):
            ax = axes[i, j]
            stats = cells[(workload, key)]
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
                ax.set_title(title, pad=6, linespacing=1.25)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=11)
            else:
                ax.tick_params(axis="y", labelleft=False)
            # Bottom-row bars reach the top of the axes, so those latencies
            # sit just above the frame. The top row still fits inside.
            ax.text(
                0.02,
                0.98 if i == 0 else 1.03,
                _latency_text(qv, aker),
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

    print("workload\tcondition\tsystem\thit\trecall_hits\tinsert_ms\tdelete_ms")
    for workload, _ in ROWS:
        for key, _ in COLUMNS:
            for system in ("qvcache", "aker"):
                s = cells[(workload, key)][system]
                rec = s["recall"]
                rec_s = f"{rec:.4f}" if np.isfinite(rec) else "nan"
                hit_s = f"{s['hit']:.4f}" if np.isfinite(s["hit"]) else "nan"
                print(
                    f"{workload}\t{key}\t{system}\t{hit_s}\t{rec_s}\t"
                    f"{_fmt_ms(s['insert_ms'])}\t{_fmt_ms(s['delete_ms'])}"
                )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qv-insert-delete",
        type=Path,
        default=repo / "logs/simZipf_refresh/qvcache/qvcache_refresh_simzipf_penalty_l1r2_d25.log",
    )
    parser.add_argument(
        "--aker-insert-delete",
        type=Path,
        default=repo / "logs/simZipf_refresh/aker/aker_refresh_simzipf_d25.log",
    )
    parser.add_argument(
        "--qv-delete-only",
        type=Path,
        default=repo / "logs/simZipf_delonly/qvcache/qvcache_refresh_delonly_simZipf_D0.25.log",
    )
    parser.add_argument(
        "--aker-delete-only",
        type=Path,
        default=repo / "logs/simZipf_delonly/aker/aker_refresh_delonly_simZipf_D0.25.log",
    )
    parser.add_argument(
        "--qv-plus-insert-delete",
        type=Path,
        default=repo
        / "logs/simZipf+_refresh/qvcache/qvcache_refresh_simzipfplus_penalty_l1r2_d25.log",
    )
    parser.add_argument(
        "--aker-plus-insert-delete",
        type=Path,
        default=repo / "logs/simZipf+_refresh/aker/aker_refresh_simzipfplus_d25.log",
    )
    parser.add_argument(
        "--qv-plus-delete-only",
        type=Path,
        default=repo / "logs/qvcache_refresh_delonly_simZipf+_D0.25.log",
    )
    parser.add_argument(
        "--aker-plus-delete-only",
        type=Path,
        default=repo / "logs/aker_refresh_delonly_simZipf+_D0.25.log",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo / "logs" / "sim_refresh_insert_vs_delete_bars.png",
    )
    args = parser.parse_args()
    plot(
        {
            ("simZipf", "insert_delete"): _pair(args.qv_insert_delete, args.aker_insert_delete),
            ("simZipf", "delete_only"): _pair(args.qv_delete_only, args.aker_delete_only),
            ("simZipf+", "insert_delete"): _pair(
                args.qv_plus_insert_delete, args.aker_plus_insert_delete
            ),
            ("simZipf+", "delete_only"): _pair(
                args.qv_plus_delete_only, args.aker_plus_delete_only
            ),
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
