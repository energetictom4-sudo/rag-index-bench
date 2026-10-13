# -*- coding: utf-8 -*-
"""磁盘倒排 IVF 的部署形态测量（配合 disk_inverted_ivf.py 使用）

模拟真实部署：新进程启动，只从磁盘加载质心、簇偏移表与 id 映射表
（共约 2 MB），向量本体留在磁盘经 memmap 按需读取。测量进程内存与
纯检索延迟（首查=冷启动读盘，后续=页缓存热态）。

用法：
  python experiments/disk_inverted_serve.py     # 前提：先跑过 disk_inverted_ivf.py 建索引
"""

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
DISK_VECS = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "disk_inverted_vecs.bin")
DISK_META = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "disk_inverted_meta.npz")

NPROBE = 32
N_QUERIES = 20
DIM = 1024
K = 10


def rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> None:
    print("=" * 70)
    print("磁盘倒排 IVF 部署形态（新进程，只加载磁盘索引）")
    print("=" * 70)

    meta = np.load(DISK_META)
    offsets = meta["offsets"]
    counts = meta["counts"]
    centroids = meta["centroids"]
    order = meta["order"]
    nlist = len(counts)
    print(f"已加载元信息：{nlist} 簇，质心 {centroids.shape}，"
          f"id 映射 {len(order)} 条（内存 ~2 MB 量级）")

    rng = np.random.default_rng(42)          # 与 disk_inverted_ivf.py 相同的查询
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    faiss.normalize_L2(queries)

    rss_before = rss_mb()
    vecs = np.memmap(DISK_VECS, dtype=np.float32, mode="r", shape=(offsets[-1], DIM))

    def search(qs):
        q = qs.astype(np.float32).copy()
        faiss.normalize_L2(q)
        probe = np.argsort(-(q @ centroids.T), axis=1)[:, :NPROBE]
        out = np.empty((q.shape[0], K), dtype=np.int64)
        for i in range(q.shape[0]):
            seg_sims, seg_gids = [], []
            for cid in probe[i]:
                s, e = offsets[cid], offsets[cid] + counts[cid]
                if e > s:
                    seg_sims.append(vecs[s:e] @ q[i])
                    seg_gids.append(order[s:e])
            sims = np.concatenate(seg_sims)
            gids = np.concatenate(seg_gids)
            out[i] = gids[np.argsort(-sims)[:K]]
        return out

    # 冷启动首查（未命中页缓存，真实读盘）
    t0 = time.perf_counter()
    search(queries)
    cold_ms = (time.perf_counter() - t0) / N_QUERIES * 1000
    rss_cold = rss_mb()

    # 热态（页缓存已驻留，后续查询）
    t0 = time.perf_counter()
    for _ in range(3):
        search(queries)
    warm_ms = (time.perf_counter() - t0) / 3 / N_QUERIES * 1000
    rss_warm = rss_mb()

    print(f"\n内存：加载元信息后 {rss_before:.0f} MB → 冷启动首查后 {rss_cold:.0f} MB"
          f" → 热态 {rss_warm:.0f} MB")
    print(f"延迟：冷启动首查 {cold_ms:.2f} ms/查询（含读盘）→ "
          f"热态 {warm_ms:.2f} ms/查询（页缓存命中）")
    print("\n对照（disk_inverted_ivf.py 同机实测）：")
    print("  全内存暴力：981 MB / 7.01 ms；全内存 IVF：1477 MB / 2.00 ms")
    if rss_before:
        print(f"\n部署形态结论：进程内存从 ~1.5 GB（全内存 IVF）降到 "
              f"{rss_warm or rss_cold:.0f} MB 量级（向量本体在磁盘，按 nprobe 簇按需换页），"
              f"热态延迟 {warm_ms:.2f} ms 与全内存 IVF 同量级。")


if __name__ == "__main__":
    main()
