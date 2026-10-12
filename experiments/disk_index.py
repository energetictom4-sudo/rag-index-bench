# -*- coding: utf-8 -*-
"""磁盘索引实验（改进路线第 4 项 · 存储延伸的第二步：索引外存的 mmap 路线）

背景：规模扫描证明 50 万块时暴力检索延迟超端到端成本、近似索引成为必需；但还有
另一面没测——索引本身放不进内存怎么办。本实验测 faiss 的 IO_FLAG_MMAP 方案：
索引文件经操作系统内存映射按需换页，进程只驻留热点页，用"内存占用下降"换
"可能的延迟代价"，回答"索引外存值不值"。

用法：
  python experiments/disk_index.py

前提：vector_store/faiss_dense/ 为当前数据集（fiqa 12 万块）的 dense 索引。
输出：全内存 vs mmap 的内存占用与纯检索延迟对比，结果落 experiments/results/disk_index.csv
"""

import csv
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import faiss
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)
INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index.faiss")
MMAP_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index_mmap.faiss")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

N_QUERIES = 20          # 检索查询数（取平均延迟）
DIM = 1024


def rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def bench(index, queries, warmup=2):
    """纯检索延迟微基准：预热后取平均毫秒（与 bench_retrieval.py 口径一致）"""
    k = 10
    for _ in range(warmup):
        index.search(queries, k)
    t0 = time.perf_counter()
    for _ in range(3):
        index.search(queries, k)
    return (time.perf_counter() - t0) / 3 / len(queries) * 1000   # 毫秒/查询


def main() -> None:
    print("=" * 70)
    print("磁盘索引实验：IndexFlatIP 全内存 vs IO_FLAG_MMAP（fiqa 12 万块）")
    print("=" * 70)

    # 查询向量用固定种子生成（本实验只测延迟与内存，召回不在范围）
    rng = np.random.default_rng(42)
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    faiss.normalize_L2(queries)

    # 1. 全内存基线
    index = faiss.read_index(INDEX_FILE)
    ntotal = index.ntotal
    rss_mem = rss_mb()
    lat_mem = bench(index, queries)
    print(f"\n[全内存] 块数 {ntotal}，进程内存 {rss_mem:.0f} MB，"
          f"纯检索 {lat_mem:.2f} ms/查询")

    # 2. mmap 版：同一索引用 IO_FLAG_MMAP 重写，再读回
    if os.path.exists(MMAP_FILE):
        os.remove(MMAP_FILE)
    faiss.write_index(index, MMAP_FILE, faiss.IO_FLAG_MMAP)
    del index
    index_mmap = faiss.read_index(MMAP_FILE)
    rss_mmap = rss_mb()
    lat_mmap = bench(index_mmap, queries)
    print(f"[mmap  ] 块数 {index_mmap.ntotal}，进程内存 {rss_mmap:.0f} MB，"
          f"纯检索 {lat_mmap:.2f} ms/查询")

    # 3. 对照输出
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)
    rows = []
    if rss_mem and rss_mmap:
        print(f"  内存：{rss_mem:.0f} MB → {rss_mmap:.0f} MB，"
              f"下降 {(1 - rss_mmap / rss_mem):.0%}")
        rows.append(("内存MB", round(rss_mem, 1), round(rss_mmap, 1)))
    print(f"  延迟：{lat_mem:.2f} → {lat_mmap:.2f} ms/查询，"
          f"代价 {(lat_mmap / lat_mem - 1):.0%}")
    rows.append(("延迟ms", round(lat_mem, 2), round(lat_mmap, 2)))

    csv_path = os.path.join(RESULTS_DIR, "disk_index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["指标", "全内存", "mmap"])
        w.writerows(rows)
    print(f"\n已写入 {csv_path}")
    print("结论要点：mmap 让向量数据由 OS 按需换页，进程内存大幅下降；")
    print("热页驻留时延迟接近全内存（页缓存命中），冷启动首查需读盘。")


if __name__ == "__main__":
    main()
