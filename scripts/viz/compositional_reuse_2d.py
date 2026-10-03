#!/usr/bin/env python3
"""2D coverage of Aker (query balls) vs QVCache (compositional S) after 20 misses.

One cluster: 200 data vectors and 20 queries in a closed rectangle.
The 20 queries are first-time lookups (all misses). A later query at x
is a hit iff the cache's real hit predicate would accept it.

Aker (standard mode): union of balls B(q_i, r_1(q_i)/4).
QVCache: r_k^S(x) <= theta[R(x)], S = union of exact top-k of the 20
queries, R = 8 x 8 buckets on the cluster box, theta from the miss d_k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOX = (0.0, 1.0)
# World for this figure: the bottom-left cluster only.
X0, X1 = 0.165, 0.385
Y0, Y1 = 0.145, 0.365
K = 5
N_DATA = 200
N_QUERIES = 20
BUCKETS = 8
GRID = 280
SEED = 7


def clip(p: np.ndarray) -> np.ndarray:
    out = np.array(p, dtype=float, copy=True)
    out[:, 0] = np.clip(out[:, 0], X0, X1)
    out[:, 1] = np.clip(out[:, 1], Y0, Y1)
    return out


def knn(data: np.ndarray, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    d2 = np.sum((data - q) ** 2, axis=1)
    idx = np.argpartition(d2, kth=k - 1)[:k]
    idx = idx[np.argsort(d2[idx])]
    return idx, np.sqrt(d2[idx])


def region_key(xy: np.ndarray, buckets: int = BUCKETS) -> tuple[int, int]:
    u = (xy[0] - X0) / (X1 - X0)
    v = (xy[1] - Y0) / (Y1 - Y0)
    bx = int(np.clip(u * buckets, 0, buckets - 1))
    by = int(np.clip(v * buckets, 0, buckets - 1))
    return bx, by


def make_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    data = rng.normal(loc=[0.275, 0.255], scale=0.042, size=(N_DATA, 2))
    data = clip(data)
    queries = clip(
        np.array(
            [
                [0.20, 0.22],
                [0.30, 0.24],
                [0.24, 0.33],
                [0.33, 0.31],
                [0.27, 0.18],
                [0.22, 0.28],
                [0.29, 0.29],
                [0.25, 0.21],
                [0.31, 0.20],
                [0.19, 0.26],
                [0.18, 0.18],
                [0.35, 0.22],
                [0.21, 0.32],
                [0.28, 0.34],
                [0.34, 0.17],
                [0.26, 0.26],
                [0.17, 0.30],
                [0.32, 0.27],
                [0.23, 0.17],
                [0.36, 0.33],
            ]
        )
    )
    return data, queries


def aker_radii(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    taus = np.empty(len(queries))
    for i, q in enumerate(queries):
        _, dists = knn(data, q, k)
        taus[i] = dists[0] / 4.0
    return taus


def admit_s(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    tags = []
    for q in queries:
        idx, _ = knn(data, q, k)
        tags.extend(idx.tolist())
    return np.array(sorted(set(tags)), dtype=int)


def learn_theta(
    data: np.ndarray, queries: np.ndarray, k: int
) -> dict[tuple[int, int], float]:
    theta: dict[tuple[int, int], float] = {}
    p = 0.90
    for q in queries:
        _, dists = knn(data, q, k)
        dk = float(dists[k - 1])
        r = region_key(q)
        if r not in theta:
            theta[r] = dk
        else:
            theta[r] = p * dk + (1.0 - p) * theta[r]
    return theta


def aker_hit(x: np.ndarray, queries: np.ndarray, taus: np.ndarray) -> bool:
    d = np.sqrt(np.sum((queries - x) ** 2, axis=1))
    return bool(np.any(d <= taus))


def qvcache_hit(
    x: np.ndarray,
    s_pts: np.ndarray,
    theta: dict[tuple[int, int], float],
    k: int,
) -> bool:
    r = region_key(x)
    if r not in theta or len(s_pts) < k:
        return False
    _, dists = knn(s_pts, x, k)
    return float(dists[k - 1]) <= theta[r]


def coverage_maps(
    data: np.ndarray,
    queries: np.ndarray,
    taus: np.ndarray,
    s_idx: np.ndarray,
    theta: dict[tuple[int, int], float],
    k: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(X0, X1, n)
    ys = np.linspace(Y0, Y1, n)
    xx, yy = np.meshgrid(xs, ys)
    aker = np.zeros((n, n), dtype=bool)
    qvc = np.zeros((n, n), dtype=bool)
    s_pts = data[s_idx]
    for i in range(n):
        for j in range(n):
            x = np.array([xx[i, j], yy[i, j]])
            aker[i, j] = aker_hit(x, queries, taus)
            qvc[i, j] = qvcache_hit(x, s_pts, theta, k)
    return xs, ys, aker, qvc


def plot_panels(
    data: np.ndarray,
    queries: np.ndarray,
    s_idx: np.ndarray,
    taus: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    aker: np.ndarray,
    qvc: np.ndarray,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    miss = "#d73027"
    hit = "#1a9850"
    cmap = ListedColormap([miss, hit])
    s_pts = data[s_idx]
    rest = np.ones(len(data), dtype=bool)
    rest[s_idx] = False

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), layout="constrained")
    titles = ("(a) Aker", "(b) QVCache")

    def scatter_data(ax, pts, filled: bool, z: int) -> None:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=16,
            marker="o",
            facecolors="black" if filled else "none",
            edgecolors="black",
            linewidths=0.7,
            zorder=z,
        )

    def scatter_queries(ax) -> None:
        ax.scatter(
            queries[:, 0],
            queries[:, 1],
            s=28,
            marker="^",
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )

    def draw(ax, m, aker_panel: bool) -> None:
        ax.set_facecolor(miss)
        if not aker_panel:
            ax.pcolormesh(
                xs,
                ys,
                m.astype(float),
                cmap=cmap,
                shading="nearest",
                vmin=0,
                vmax=1,
                rasterized=True,
            )
        scatter_data(ax, data[rest], filled=False, z=3)
        scatter_data(ax, s_pts, filled=True, z=4)
        if aker_panel:
            for q, tau in zip(queries, taus):
                ax.add_patch(
                    Circle(
                        q,
                        tau,
                        facecolor=hit,
                        edgecolor="#145a32",
                        lw=0.8,
                        zorder=8,
                        clip_on=True,
                    )
                )
        scatter_queries(ax)
        ax.set_aspect("equal")
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y0, Y1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    draw(axes[0], aker, True)
    draw(axes[1], qvc, False)
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    legend = [
        Patch(facecolor=hit, edgecolor="black", linewidth=0.6, label="Hit Region"),
        Patch(facecolor=miss, edgecolor="black", linewidth=0.6, label="Miss Region"),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=7,
            label="Data",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label="Cached Data",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="^",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="Query",
        ),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def dump_canvas_payload(
    data: np.ndarray,
    queries: np.ndarray,
    s_idx: np.ndarray,
    taus: np.ndarray,
    aker: np.ndarray,
    qvc: np.ndarray,
    out: Path,
) -> None:
    # Downsample maps for SVG (every 3rd cell).
    step = 3
    a_small = aker[::step, ::step]
    q_small = qvc[::step, ::step]
    n = a_small.shape[0]

    def pack(m: np.ndarray) -> str:
        bits = m.reshape(-1).astype(np.uint8)
        packed = np.packbits(bits)
        return packed.tobytes().hex()

    payload = {
        "box": [X0, X1, Y0, Y1],
        "k": K,
        "n": int(n),
        "data": data.round(4).tolist(),
        "queries": queries.round(4).tolist(),
        "s_idx": s_idx.tolist(),
        "taus": taus.round(5).tolist(),
        "aker_bits": pack(a_small),
        "qvc_bits": pack(q_small),
        "aker_hit_frac": float(aker.mean()),
        "qvc_hit_frac": float(qvc.mean()),
        "s_size": int(len(s_idx)),
    }
    out.write_text(json.dumps(payload, separators=(",", ":")))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("papers/figures/compositional_reuse_2d.png"))
    p.add_argument("--json", type=Path, default=Path("papers/figures/compositional_reuse_2d.json"))
    args = p.parse_args()

    rng = np.random.default_rng(SEED)
    data, queries = make_scene(rng)
    assert len(data) == N_DATA and len(queries) == N_QUERIES
    taus = aker_radii(data, queries, K)
    s_idx = admit_s(data, queries, K)
    theta = learn_theta(data, queries, K)
    xs, ys, aker, qvc = coverage_maps(data, queries, taus, s_idx, theta, K, GRID)

    plot_panels(data, queries, s_idx, taus, xs, ys, aker, qvc, args.out)
    dump_canvas_payload(data, queries, s_idx, taus, aker, qvc, args.json)

    print(f"data={len(data)} queries={len(queries)} k={K}")
    print(f"|S|={len(s_idx)}  learned cells={len(theta)}")
    print(f"Aker tau mean={taus.mean():.4f}  min={taus.min():.4f}  max={taus.max():.4f}")
    print(f"Aker hit area={aker.mean():.3f}  QVCache hit area={qvc.mean():.3f}")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.pdf')}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
