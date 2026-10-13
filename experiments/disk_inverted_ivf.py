# -*- coding: utf-8 -*-
"""磁盘倒排 IVF 实验（改进路线第 4 项 · 存储延伸的第三步：索引外存的正确形态）

背景：mmap 实验证明暴力检索全量扫描使按需换页失效（内存不降）。本实验实现
"磁盘倒排表"：内存只驻留质心与簇偏移表，向量按簇连续落盘（np.memmap 按需读取），
检索时只读 nprobe 个簇的段——访问天然稀疏，与微软 DiskANN 的思路同族。

对照三形态（fiqa 12 万块，nlist=256 与 indexes/ivf.py 一致）：
  ① 全内存暴力（IndexFlatIP）
  ② 全内存 IVF（faiss IndexIVFFlat，nprobe=32）
  ③ 磁盘倒排 IVF（本实验实现，nprobe=32）
测：进程内存、纯检索延迟、检索质量（与 ② 逐位一致验证正确性）。

用法：
  python experiments/disk_inverted_ivf.py

前提：vector_store/faiss_dense/ 为当前数据集（fiqa）的 dense 索引。
输出：experiments/results/disk_inverted_ivf.csv 与对比汇总。
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
DISK_VECS = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "disk_inverted_vecs.bin")
DISK_META = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "disk_inverted_meta.npz")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

NLIST = 256              # 与 indexes/ivf.py 的 k-means 簇数一致
NPROBE = 32              # 检索簇数（甜蜜点档位）
N_QUERIES = 20
DIM = 1024
K = 10


def rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def bench(search_fn, queries, warmup=2):
    """纯检索延迟微基准：预热后 3 轮取平均，返回毫秒/查询"""
    for _ in range(warmup):
        search_fn(queries)
    t0 = time.perf_counter()
    for _ in range(3):
        search_fn(queries)
    return (time.perf_counter() - t0) / 3 / len(queries) * 1000


def build_disk_inverted(vecs, centroids, assign):
    """把向量按簇重排后连续落盘，返回 (offsets, counts, order)

    磁盘布局：disk_inverted_vecs.bin = 按簇 id 升序连续排列的 float32 向量，
    簇 i 的向量在 [offsets[i], offsets[i]+counts[i]) 区间（np.memmap 按段读取）。
    order = 磁盘行号 → 原始向量 id 的映射表（磁盘索引的标准组成，随 meta 落盘）。
    """
    order = np.argsort(assign, kind="stable")          # 按簇号重排的下标
    sorted_vecs = vecs[order]                          # (n, dim) 重排（内存一次性拷贝）
    sorted_vecs.tofile(DISK_VECS)
    counts = np.bincount(assign, minlength=NLIST).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    np.savez(DISK_META, offsets=offsets, counts=counts,
             centroids=centroids, order=order)
    return offsets, counts, order


def make_disk_searcher(offsets, counts, centroids, order):
    """返回磁盘倒排检索函数：质心与 id 映射表全在内存，向量段经 memmap 按需读

    返回的 id 为原始向量 id（经 order 映射），与 faiss IVF 的返回语义一致。
    """
    vecs = np.memmap(DISK_VECS, dtype=np.float32, mode="r", shape=(offsets[-1], DIM))

    def search(queries):
        q = queries.astype(np.float32).copy()
        faiss.normalize_L2(q)
        # 1. 查询与质心算内积 → 取 nprobe 个最近簇
        sim_qc = q @ centroids.T          # (nq, nlist)
        probe = np.argsort(-sim_qc, axis=1)[:, :NPROBE]
        all_ids = np.empty((q.shape[0], K), dtype=np.int64)
        for i in range(q.shape[0]):
            # 2. 只读命中的簇段（memmap 按页读取，页缓存命中时零 IO），
            #    磁盘行号经 order 映射回原始向量 id
            seg_sims, seg_gids = [], []
            for cid in probe[i]:
                s, e = offsets[cid], offsets[cid] + counts[cid]
                if e > s:
                    seg = vecs[s:e]                  # 该簇向量段
                    seg_sims.append(seg @ q[i])
                    seg_gids.append(order[s:e])
            sims = np.concatenate(seg_sims)
            gids = np.concatenate(seg_gids)
            # 3. nprobe 簇内的精确 top-k（与 faiss IVF nprobe=32 的语义一致）
            all_ids[i] = gids[np.argsort(-sims)[:K]]
        return all_ids

    return search, vecs


def main() -> None:
    print("=" * 70)
    print("磁盘倒排 IVF 实验（fiqa 12 万块，nlist=256 / nprobe=32）")
    print("=" * 70)

    block_index = faiss.read_index(INDEX_FILE)
    n = block_index.ntotal
    vecs = block_index.reconstruct_n(0, n)            # (n, dim) float32，已 L2 归一化
    print(f"块向量已加载：{n} × {DIM}")

    rng = np.random.default_rng(42)
    queries = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    faiss.normalize_L2(queries)

    # ---------- ① 全内存暴力 ----------
    lat_flat = bench(lambda q: block_index.search(q, K), queries)
    rss_flat = rss_mb()
    print(f"\n[① 全内存暴力] 内存 {rss_flat:.0f} MB，延迟 {lat_flat:.2f} ms/查询")

    # ---------- ② 全内存 IVF（faiss） ----------
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(DIM), DIM, NLIST,
                             faiss.METRIC_INNER_PRODUCT)
    ivf.nprobe = NPROBE
    ivf.train(vecs)
    ivf.add(vecs)
    # 磁盘倒排必须与 ② 共用同一套质心与簇分配，检索结果才能逐位一致
    centroids = np.ascontiguousarray(ivf.quantizer.reconstruct_n(0, NLIST))
    faiss.normalize_L2(centroids)
    _, assign = ivf.quantizer.search(vecs, 1)
    assign = assign.ravel()
    lat_ivf = bench(lambda q: ivf.search(q, K), queries)
    rss_ivf = rss_mb()
    ids_faiss = ivf.search(queries, K)[1]     # 参考结果：释放 ivf 前保存，供正确性验证
    print(f"[② 全内存 IVF]  内存 {rss_ivf:.0f} MB，延迟 {lat_ivf:.2f} ms/查询")

    # ---------- ③ 磁盘倒排 IVF（本实验实现） ----------
    offsets, counts, order = build_disk_inverted(vecs, centroids, assign)
    # 释放全量向量与内存版索引，让 ③ 的内存测量只反映"质心+偏移表+页缓存"形态
    del block_index, vecs, ivf
    import gc
    gc.collect()
    searcher, mmap_obj = make_disk_searcher(offsets, counts, centroids, order)
    lat_disk = bench(searcher, queries)
    rss_disk = rss_mb()
    print(f"[③ 磁盘倒排 IVF] 内存 {rss_disk:.0f} MB，延迟 {lat_disk:.2f} ms/查询"
          f"（同进程先后测量，含前序形态的内存残留；部署形态见 disk_inverted_serve.py）")

    # ---------- 正确性验证：③ 与 ② 检索质量对比 ----------
    # 同一质心、同一簇分配、同一 nprobe 下，③ 与 ② 的 top-10 应逐位一致
    ids_disk = searcher(queries)
    identical = sum(np.array_equal(a, b) for a, b in zip(ids_faiss, ids_disk))
    overlap = sum(len(set(a) & set(b)) for a, b in zip(ids_faiss, ids_disk)) / (N_QUERIES * K)
    print(f"\n[正确性验证] 磁盘倒排与 faiss IVF(nprobe={NPROBE}) 的 top-10 平均重叠："
          f"{overlap:.1%}（{identical}/{N_QUERIES} 个查询逐位一致）")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print("三形态对比汇总")
    print("=" * 70)
    rows = [
        ("形态", "进程内存(MB)", "纯检索延迟(ms/查询)"),
        ("全内存暴力", round(rss_flat or 0, 1), round(lat_flat, 2)),
        ("全内存IVF", round(rss_ivf or 0, 1), round(lat_ivf, 2)),
        ("磁盘倒排IVF", round(rss_disk or 0, 1), round(lat_disk, 2)),
    ]
    for r in rows:
        print(f"  {r[0]:<10} {r[1]:>14} {r[2]:>20}")
    if rss_flat and rss_disk:
        print(f"\n  内存：磁盘倒排比全内存暴力低 {rss_flat - rss_disk:.0f} MB"
              f"（{rss_flat / max(rss_disk, 1):.0f} 倍下降）")
    print(f"  延迟：磁盘倒排 vs 全内存 IVF = {lat_disk:.2f} vs {lat_ivf:.2f} ms"
          f"（热态页缓存命中，代价 {(lat_disk / lat_ivf - 1):.0%}）")
    disk_size = os.path.getsize(DISK_VECS) / (1024 * 1024)
    print(f"  磁盘文件：{disk_size:.1f} MB（向量本体）+ 质心/偏移表内存 ~"
          f"{NLIST * DIM * 4 / 1024:.0f} KB")

    csv_path = os.path.join(RESULTS_DIR, "disk_inverted_ivf.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"\n已写入 {csv_path}")


if __name__ == "__main__":
    main()
