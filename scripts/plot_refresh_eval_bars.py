#!/usr/bin/env python3
"""Bar charts of eval-phase hit rate and hit recall from refresh logs.

Each run directory is expected to contain qvcache.log and aker.log. The script
reads the refresh_eval summary from each log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUNS = (
    "logs/refresh/sphere-1m/sim-100k-0.99/del0.05",
    "logs/refresh/sphere-1m/sim-100k-0.99/del0.15",
    "logs/refresh/sphere-1m/sim-100k-0.99/del0.25",
    "logs/refresh/sphere-1m/simplus-100k-0.99/del0.05",
    "logs/refresh/sphere-1m/simplus-100k-0.99/del0.15",
    "logs/refresh/sphere-1m/simplus-100k-0.99/del0.25",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workload_name(path: Path) -> str:
    workload = path.parent.name
    for suffix in ("-100k-0.99", "-100k"):
        if workload.endswith(suffix):
            workload = workload[: -len(suffix)]
            break
    if workload == "sim":
        return "simZipf"
    if workload == "simplus":
        return "simZipf+"
    return workload


def delete_value(path: Path) -> float:
    name = path.name[3:] if path.name.startswith("del") else path.name
    try:
        return float(name)
    except ValueError:
        return 0.0


def delete_label(path: Path) -> str:
    name = path.name[3:] if path.name.startswith("del") else path.name
    return f"del {name}"


def load_eval(path: Path) -> tuple[float, float]:
    with path.open() as handle:
        for line in handle:
            start = line.find('{"event": "refresh_eval"')
            if start < 0:
                continue
            event = json.loads(line[start:])
            return float(event["hit_ratio"]), float(event["recall_hits"])
    raise SystemExit(f"no refresh_eval event in {path}")


def collect(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"not a directory: {run_dir}")
        row = {
            "workload": workload_name(run_dir),
            "delete": delete_value(run_dir),
            "label": delete_label(run_dir),
        }
        found = False
        for cache in ("qvcache", "aker"):
            log_path = run_dir / f"{cache}.log"
            if not log_path.is_file():
                continue
            hit_rate, hit_recall = load_eval(log_path)
            row[cache] = {"hit_rate": hit_rate, "hit_recall": hit_recall}
            found = True
        if not found:
            raise SystemExit(f"no qvcache.log or aker.log in {run_dir}")
        rows.append(row)
    return rows


def grouped_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    order = []
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        workload = row["workload"]
        if workload not in buckets:
            order.append(workload)
            buckets[workload] = []
        buckets[workload].append(row)
    preferred = ["simZipf", "simZipf+"]
    order.sort(key=lambda name: preferred.index(name) if name in preferred else len(preferred))
    return [(name, sorted(buckets[name], key=lambda row: row["delete"])) for name in order]


def plot(rows: list[dict], output: Path) -> None:
    groups = grouped_rows(rows)
    deletes = sorted({row["delete"] for row in rows})
    by_key = {(row["workload"], row["delete"]): row for row in rows}
    caches = [cache for cache in ("qvcache", "aker") if any(cache in row for row in rows)]
    metrics = (
        ("hit_rate", "Hit rate"),
        ("hit_recall", "Hit recall"),
    )
    width = 0.36 if len(caches) > 1 else 0.5
    offsets = [((i - (len(caches) - 1) / 2) * width) for i in range(len(caches))]
    colors = {"qvcache": "#d97706", "aker": "#2563eb"}
    names = {"qvcache": "QVCache", "aker": "Aker"}
    x = list(range(len(metrics)))

    fig, axes = plt.subplots(
        len(groups),
        len(deletes),
        figsize=(3.6 * len(deletes), 3.5 * len(groups)),
        sharey=True,
        squeeze=False,
    )
    legend_handles = None
    legend_labels = None
    for row_axes, (workload, _group) in zip(axes, groups):
        for ax, delete in zip(row_axes, deletes):
            row = by_key.get((workload, delete))
            for cache, offset in zip(caches, offsets):
                present = row is not None and cache in row
                values = [
                    row[cache][metric] * 100 if present else 0.0
                    for metric, _title in metrics
                ]
                bars = ax.bar(
                    [xi + offset for xi in x],
                    values,
                    width=width,
                    label=names[cache],
                    color=colors[cache],
                )
                for bar, value in zip(bars, values):
                    if not present:
                        bar.set_height(0)
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 1.2,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            rate = f"{delete:g}"
            ax.set_title(f"{workload} del {rate}")
            ax.set_xlim(-0.5, len(metrics) - 0.5)
            ax.set_xticks(x, [title for _metric, title in metrics])
            ax.set_ylim(0, 118)
            ax.axhline(100, color="#d4d4d8", linewidth=0.6, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        row_axes[0].set_ylabel("Percent")

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(caches), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Run directories containing qvcache.log and aker.log. "
        "Defaults to the sphere-1m simZipf and simZipf+ logs at del 0.05, 0.15, and 0.25.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "logs/refresh/sphere-1m/eval_hit_bars.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    root = repo_root()
    run_dirs = [path if path.is_absolute() else root / path for path in args.runs]
    if not run_dirs:
        run_dirs = [root / path for path in DEFAULT_RUNS]
    output = args.out if args.out.is_absolute() else root / args.out
    rows = collect(run_dirs)
    plot(rows, output)
    for row in rows:
        label = f"{row['workload']} {row['label']}"
        parts = []
        for cache in ("qvcache", "aker"):
            if cache not in row:
                continue
            parts.append(
                f"{cache} hit rate {row[cache]['hit_rate']:.4f} "
                f"hit recall {row[cache]['hit_recall']:.4f}"
            )
        print(f"{label}: " + "; ".join(parts), file=sys.stdout)
    print(output)


if __name__ == "__main__":
    main()
