#!/usr/bin/env python3
"""Generate the conceptual figures for the QVCache/Aker positioning section.

Produces two figures:

  fig_reuse_geometry      Atomic (result-materialized) vs. compositional
                          (data-materialized) reuse, drawn on a real 2D point
                          cloud so the radii and coverage are measured rather
                          than hand-placed.
  fig_threshold_dynamics  Drift of Aker's per-query reuse radius under its
                          published update rule, as a function of the fraction
                          of hits that are bit-exact repeats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Aker's published defaults (Eq. 1 and Sec. 4.3).
ALPHA_LOOSEN = 1.1
ALPHA_TIGHTEN = 0.25
TAU_INIT_FRACTION = 0.25

GRAY = "#b8bcc4"
BLUE = "#2b6cb0"
ORANGE = "#c05621"
RED = "#c53030"
GREEN = "#276749"


def knn(points: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest points to query, ascending by distance."""
    dists = np.linalg.norm(points - query, axis=1)
    return np.argsort(dists)[:k]


def radius(points: np.ndarray, query: np.ndarray, j: int) -> float:
    """Distance from query to its j-th nearest neighbour (1-indexed)."""
    dists = np.sort(np.linalg.norm(points - query, axis=1))
    return float(dists[j - 1])


def build_scenario(seed: int, n_points: int, k: int) -> dict:
    """Place two cached queries whose result balls overlap, plus a query
    sitting in the interstice between their reuse balls."""
    rng = np.random.default_rng(seed)
    points = np.vstack(
        [
            rng.normal([0.0, 0.0], 0.62, size=(int(n_points * 0.5), 2)),
            rng.normal([1.75, 0.70], 0.34, size=(int(n_points * 0.3), 2)),
            rng.normal([0.70, -1.70], 0.85, size=(int(n_points * 0.2), 2)),
        ]
    )

    centre = np.array([0.55, 0.25])
    direction = np.array([1.0, 0.45])
    direction = direction / np.linalg.norm(direction)

    # Separate the two cached queries by a multiple of the local top-k radius
    # so that their result balls overlap, then take the midpoint as the new
    # query. Among separations that keep coverage high and draw the answer
    # from both cached results, prefer the widest so the two are legible.
    best = None
    for multiple in np.linspace(0.8, 1.8, 41):
        scale = radius(points, centre, k)
        q1 = centre - 0.5 * multiple * scale * direction
        q2 = centre + 0.5 * multiple * scale * direction
        query = 0.5 * (q1 + q2)

        a1, a2 = knn(points, q1, k), knn(points, q2, k)
        cached = np.union1d(a1, a2)
        truth = knn(points, query, k)
        coverage = len(np.intersect1d(truth, cached)) / k

        # Answer computed over the cached subset only.
        local = cached[np.argsort(np.linalg.norm(points[cached] - query, axis=1))][:k]
        only1 = len(np.setdiff1d(np.intersect1d(local, a1), a2))
        only2 = len(np.setdiff1d(np.intersect1d(local, a2), a1))

        if coverage < 0.85 or only1 == 0 or only2 == 0:
            continue
        score = (multiple,)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "points": points,
                "q1": q1,
                "q2": q2,
                "query": query,
                "a1": a1,
                "a2": a2,
                "cached": cached,
                "truth": truth,
                "local": local,
                "coverage": coverage,
                "r1": radius(points, q1, k),
                "r2": radius(points, q2, k),
                "tau1": TAU_INIT_FRACTION * radius(points, q1, 1),
                "tau2": TAU_INIT_FRACTION * radius(points, q2, 1),
                "rq": radius(points, query, k),
                "k": k,
            }
    assert best is not None
    return best


def plot_reuse_geometry(scn: dict, out_dir: Path) -> None:
    points, k = scn["points"], scn["k"]
    q1, q2, query = scn["q1"], scn["q2"], scn["query"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))

    for ax in axes:
        ax.scatter(points[:, 0], points[:, 1], s=9, c=GRAY, zorder=1,
                   linewidths=0, label="database vectors $P$")
        for q, marker in ((q1, "X"), (q2, "X")):
            ax.scatter(*q, s=70, c="black", marker=marker, zorder=5)
        ax.scatter(*query, s=150, c=RED, marker="*", zorder=6,
                   edgecolors="white", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 2.55)
        ax.set_ylim(-1.55, 2.05)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- Panel (a): result-materialized -------------------------------
    ax = axes[0]
    for q, tau in ((q1, scn["tau1"]), (q2, scn["tau2"])):
        ax.add_patch(Circle(q, tau, facecolor=RED, alpha=0.30,
                            edgecolor=RED, lw=1.2, zorder=4))
    ax.annotate(r"$B(q_1,\tau_1)$", xy=q1, xytext=(q1[0] - 0.95, q1[1] + 1.05),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate(r"$B(q_2,\tau_2)$", xy=q2, xytext=(q2[0] + 0.12, q2[1] + 1.15),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate("$q$ outside every\nreuse ball: miss", xy=query,
                xytext=(query[0] - 1.25, query[1] - 1.25), fontsize=9,
                color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.set_title("(a) result-materialized: atomic reuse", fontsize=10)

    # ---- Panel (b): data-materialized ---------------------------------
    ax = axes[1]
    ax.add_patch(Circle(q1, scn["r1"], facecolor=BLUE, alpha=0.11,
                        edgecolor=BLUE, lw=1.0, ls="--", zorder=2))
    ax.add_patch(Circle(q2, scn["r2"], facecolor=ORANGE, alpha=0.11,
                        edgecolor=ORANGE, lw=1.0, ls="--", zorder=2))
    ax.scatter(points[scn["a1"], 0], points[scn["a1"], 1], s=26, c=BLUE,
               zorder=3, linewidths=0, label="fetched for $q_1$")
    ax.scatter(points[scn["a2"], 0], points[scn["a2"], 1], s=26, c=ORANGE,
               zorder=3, linewidths=0, label="fetched for $q_2$")
    ax.add_patch(Circle(query, scn["rq"], facecolor="none", edgecolor=GREEN,
                        lw=1.6, zorder=4))
    ax.scatter(points[scn["local"], 0], points[scn["local"], 1], s=95,
               facecolors="none", edgecolors=GREEN, lw=1.3, zorder=5,
               label=r"returned $\mathrm{kNN}_S(q)$")
    n_from_1 = len(np.intersect1d(scn["local"], scn["a1"]))
    n_from_2 = len(np.intersect1d(scn["local"], scn["a2"]))
    ax.annotate(
        f"$q$ answered by composing\n{n_from_1} + {n_from_2} vectors from two\n"
        f"earlier misses: hit",
        xy=query, xytext=(query[0] - 1.30, query[1] - 1.42), fontsize=9,
        color=GREEN, ha="left",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9),
    )
    ax.set_title("(b) data-materialized: compositional reuse", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9,
              handletextpad=0.4, borderpad=0.4)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_reuse_geometry.{suffix}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)

    ratio1 = scn["r1"] / scn["tau1"]
    print(f"[reuse_geometry] k={k}, n={len(points)}")
    print(f"  r_k(q1)/tau_1        = {ratio1:.1f}")
    print(f"  coverage of kNN_P(q) by A1 u A2 = {scn['coverage']:.2f}")
    print(f"  returned set drawn from A1/A2   = {n_from_1}/{n_from_2}")
    print(f"  d(q,q1)/tau_1        = "
          f"{np.linalg.norm(query - q1) / scn['tau1']:.1f}")


def plot_threshold_dynamics(out_dir: Path, n_hits: int = 240) -> None:
    """Drift of tau_q/minD under Aker's update rule for varying rates of
    bit-exact query repetition."""
    breakeven = np.log(1.0 / ALPHA_TIGHTEN) / np.log(ALPHA_LOOSEN)
    p_breakeven = breakeven / (breakeven + 1.0)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    hits = np.arange(n_hits + 1)
    fractions = [0.0, 0.50, 0.80, 0.90, p_breakeven, 0.98]

    for p in fractions:
        drift = p * np.log(ALPHA_LOOSEN) + (1.0 - p) * np.log(ALPHA_TIGHTEN)
        tau = TAU_INIT_FRACTION * np.exp(drift * hits)
        tau = np.minimum(tau, 1.0)  # capped above by minD
        if abs(p - p_breakeven) < 1e-9:
            label = rf"$p_{{\mathrm{{eq}}}}={p:.3f}$ (break-even)"
            style = dict(lw=2.0, color="black", ls="--")
        else:
            label = rf"$p_{{\mathrm{{eq}}}}={p:.2f}$"
            style = dict(lw=1.6)
        ax.plot(hits, tau, label=label, **style)

    ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
    ax.text(n_hits * 0.985, 1.18, r"cap: $\tau_q \leq \mathtt{minD}$",
            fontsize=8, color="gray", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 4.0)
    ax.set_xlim(0, n_hits)
    ax.set_xlabel("cache hits served by the entry")
    ax.set_ylabel(r"reuse radius $\tau_q\,/\,\mathtt{minD}$")
    ax.legend(fontsize=7.5, loc="lower left", ncol=2, frameon=True,
              framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_threshold_dynamics.{suffix}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)

    print("[threshold_dynamics]")
    print(f"  equality hits required per approximate hit = {breakeven:.2f}")
    print(f"  break-even exact-repetition rate           = {p_breakeven:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "papers" / "figures")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-points", type=int, default=160)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario(args.seed, args.n_points, args.k)
    plot_reuse_geometry(scenario, args.out_dir)
    plot_threshold_dynamics(args.out_dir)
    print(f"\nwrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
