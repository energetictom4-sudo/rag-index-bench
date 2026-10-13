# -*- coding: utf-8 -*-
"""IVF 簇粒度分析（改进路线第 5 项 · 机理验证的第二部分）

问题："低档位 HNSW 全面优于 IVF"的机理解释是——IVF 的 nprobe 是"整簇粒度的
粗闸门"，HNSW 按到查询的距离分配搜索预算。本脚本把前半句量化：k-means 的簇
大小天然不均衡，nprobe=1 时的实际搜索预算（覆盖向量数）随查询在极小~极大
之间剧烈波动，而 HNSW 的 ef_search 预算稳定可控。

用法：
  python experiments/analyze_ivf_clusters.py

前提：已运行 disk_inverted_ivf.py（生成簇分配 meta）。
输出：experiments/results/ivf_cluster_sizes.png（直方图）与 ivf_clusters.csv
"""

import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)
DISK_META = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "disk_inverted_meta.npz")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

SERIES = "#2a78d6"      # 参考调色板 蓝 slot 1
INK = "#52514e"         # 次级文字
GRID = "#e8e6e2"        # 弱网格


def main() -> None:
    print("=" * 70)
    print("IVF 簇粒度分析（fiqa 12 万块，nlist=256）")
    print("=" * 70)

    meta = np.load(DISK_META)
    counts = meta["counts"]
    centroids = meta["centroids"]
    n, nlist = counts.sum(), len(counts)
    print(f"向量 {n} 块 / {nlist} 簇，平均每簇 {n / nlist:.0f} 块")
    print(f"簇大小分布：最小 {counts.min()}，最大 {counts.max()}，"
          f"中位数 {np.median(counts):.0f}，标准差 {counts.std():.0f}")
    print(f"最小 10 簇覆盖 {counts[np.argsort(counts)[:10]].sum()} 块"
          f"（{counts[np.argsort(counts)[:10]].sum() / n:.1%}），"
          f"最大 10 簇覆盖 {counts[np.argsort(counts)[-10:]].sum()} 块"
          f"（{counts[np.argsort(counts)[-10:]].sum() / n:.1%}）")

    # nprobe=1 的预算波动：随机查询命中最近簇的大小分布
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((200, 1024)).astype(np.float32)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    nearest = np.argmax(q @ centroids.T, axis=1)
    probe_sizes = counts[nearest]
    print(f"\nnprobe=1 的实际搜索预算（200 个随机查询命中簇的大小）：")
    print(f"  最小 {probe_sizes.min()} 块 / 最大 {probe_sizes.max()} 块 / "
          f"平均 {probe_sizes.mean():.0f} 块——预算随查询波动 "
          f"{probe_sizes.max() / max(probe_sizes.min(), 1):.0f} 倍")
    print("  对照：HNSW ef_search=4 固定访问约 4×M 级节点数（预算稳定可控）")

    # 图：簇大小直方图（对数 x 轴，附均值/中位数标注）
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(counts, bins=40, color=SERIES, edgecolor="white", linewidth=0.5)
    for v, label in ((counts.mean(), "均值"), (np.median(counts), "中位数")):
        ax.axvline(v, color=INK, linestyle="--", linewidth=1.2)
        ax.text(v, ax.get_ylim()[1] * 0.95, f"{label} {v:.0f}", color=INK,
                ha="left", va="top", fontsize=9,
                bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xscale("log")
    ax.set_xlabel("簇内向量数（对数轴）", color=INK)
    ax.set_ylabel("簇数", color=INK)
    ax.set_title("IVF 簇大小分布（fiqa 12 万块 / 256 簇）——搜索预算的粒度",
                 fontsize=12, color="#0b0b0b")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "ivf_cluster_sizes.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"\n图已保存 {png_path}")

    csv_path = os.path.join(RESULTS_DIR, "ivf_clusters.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["簇号", "向量数"])
        for i, c in enumerate(counts):
            w.writerow([i, int(c)])
    print(f"数据已保存 {csv_path}")


if __name__ == "__main__":
    main()
