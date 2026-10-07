# -*- coding: utf-8 -*-
"""
规模扫描实验（阶段三收尾工具）：寻找"暴力检索扛不住"的交叉点

问题：dense 暴力检索延迟随语料块数线性增长，何时会成为查询瓶颈？
方法：把 dense 的 17071 块向量复制放大到 5 万~50 万块，每个规模下分别构建
      dense（IndexFlatIP 暴力）/ ivf（nprobe=32）/ hnsw（ef_search=64）三种索引，
      测量纯检索延迟（与 bench_retrieval.py 同口径）与召回率（官方口径，复制的块
      与原块 doc_id 相同，文档级判分依然有效）。
交叉点判据：暴力检索延迟超过端到端向量化成本（Ollama bge-m3 约 200 ms）时，
      检索成为查询瓶颈，近似索引从"可选优化"变为"必需"。
输出：
  - 控制台汇总表（规模 × 方案：建库耗时、延迟、加速比、召回率）
  - experiments/results/scale_scan.csv
  - experiments/results/scale_scan.png（规模-延迟双对数曲线 + 向量化成本参考线）

用法：
  python experiments/scale_scan.py
前置：faiss_dense/ 索引已建好，题库为官方口径（119 题）
注意：50 万块 × 1024 维 float32 矩阵约 2 GB，峰值内存约 4~6 GB；
      如需 100 万档位可改 SCALE_LIST（内存需求约翻倍）。
"""

import csv
import json
import os
import sys
import time

import faiss
import numpy as np

# 解决 Windows 控制台中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

# ================ 参数配置区 ================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)   # 与 rag_demo._DATA_ROOT / experiments/common.py 一致

CHUNKS_FILE = os.path.join(DATA_ROOT, "beir_chunks.json")
QUESTIONS_FILE = os.path.join(DATA_ROOT, "beir_题库.json")
DENSE_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index.faiss")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

SCALE_LIST = (50_000, 100_000, 200_000, 500_000)   # 放大后的总块数档位
NPROBE = 32       # ivf 搜索簇数（甜蜜点）
EF_SEARCH = 64    # hnsw 搜索宽度（甜蜜点）
M = 32            # hnsw 建图连接数（与 hnsw 方案一致）
EF_CONSTRUCTION = 64   # hnsw 建图搜索宽度（与 hnsw 方案一致）
TOPK = 50         # 检索返回片段数
RECALL_KS = (10, 50)          # 召回率统计档位
N_LATENCY_QUERIES = 500   # 延迟测量的随机查询数
WARMUP = 10               # 延迟测量预热次数
SEED = 42                 # 随机查询向量种子

VECTORIZE_COST_MS = 200   # 端到端向量化成本参考线（Ollama bge-m3 实测约 200ms/题）

# ======================== 以下为代码逻辑，无需改动 ========================


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_recall(index: faiss.Index, query_vecs: np.ndarray, questions: list,
                    doc_ids: list, topk: int) -> dict:
    """全部题的文档级召回率（官方口径；放大语料中复制的块与原块 doc_id 相同，判分有效）"""
    per_k_hits = {k: [] for k in RECALL_KS}
    for qi, q in enumerate(questions):
        gold_docs = {doc_ids[i] for i in q["相关块编号"]}
        if not gold_docs:
            continue
        _, row = index.search(query_vecs[qi:qi + 1], topk)
        covered = [doc_ids[int(i)] for i in row[0]]
        for k in RECALL_KS:
            hit = len(set(covered[:k]) & gold_docs)
            per_k_hits[k].append(hit / len(gold_docs))
    return {f"recall@{k}": float(np.mean(v)) for k, v in per_k_hits.items()}


def bench_latency(index: faiss.Index, queries: np.ndarray, topk: int) -> float:
    """纯检索平均延迟（与 bench_retrieval.py 同口径），返回毫秒"""
    for q in queries[:WARMUP]:
        index.search(q.reshape(1, -1), topk)
    times = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), topk)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def build_and_measure(name: str, build_fn, index, query_vecs, questions,
                      doc_ids, rand_queries, rows, scale) -> None:
    """建库计时 + 测召回与延迟 + 释放索引"""
    t0 = time.perf_counter()
    index = build_fn()
    build_time = time.perf_counter() - t0
    rec = evaluate_recall(index, query_vecs, questions, doc_ids, TOPK)
    lat = bench_latency(index, rand_queries, TOPK)
    rows.append({
        "规模(块)": scale, "方案": name, "建库秒": round(build_time, 2),
        "平均延迟ms": round(lat, 3),
        "Recall@10": round(rec["recall@10"] * 100, 2),
        "Recall@50": round(rec["recall@50"] * 100, 2),
    })
    print(f"    {name:<22} 建库 {build_time:>8.1f}s  延迟 {lat:>9.3f} ms  "
          f"Recall@50 {rec['recall@50']:>7.2%}")
    del index


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chunk_items = load_json(CHUNKS_FILE)
    base_doc_ids = [c["doc_id"] for c in chunk_items]
    questions = load_json(QUESTIONS_FILE)
    n_base = len(chunk_items)

    # 复用 ann_curve.py 缓存的 119 题查询向量（召回测量）+ 固定种子随机向量（延迟测量）
    from ann_curve import load_query_vectors
    query_vecs = load_query_vectors(questions)

    dense_base = faiss.read_index(DENSE_INDEX_FILE)
    base_matrix = np.array(dense_base.reconstruct_n(0, dense_base.ntotal), dtype=np.float32)
    d = base_matrix.shape[1]
    rng = np.random.default_rng(SEED)
    rand_queries = rng.standard_normal((N_LATENCY_QUERIES, d)).astype(np.float32)
    faiss.normalize_L2(rand_queries)
    print(f"基础语料：{n_base} 块 × {d} 维，放大档位 {SCALE_LIST}，"
          f"向量化成本参考线 {VECTORIZE_COST_MS} ms\n")

    rows = []
    for scale in SCALE_LIST:
        # 复制放大：矩阵与 doc_id 同步 tile 到目标规模（复制的块与原块 doc_id 相同）
        reps = (scale + n_base - 1) // n_base
        matrix = np.tile(base_matrix, (reps, 1))[:scale]
        doc_ids = (base_doc_ids * reps)[:scale]
        print(f"[规模 {scale:,} 块] 矩阵 {matrix.nbytes / 1024 / 1024:.0f} MB")

        def make_dense():
            idx = faiss.IndexFlatIP(d)
            idx.add(matrix)
            return idx

        def make_ivf():
            quantizer = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFFlat(quantizer, d, 256, faiss.METRIC_INNER_PRODUCT)
            idx.train(matrix)
            idx.add(matrix)
            idx.nprobe = NPROBE
            return idx

        def make_hnsw():
            idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = EF_CONSTRUCTION
            idx.add(matrix)
            idx.hnsw.efSearch = EF_SEARCH
            return idx

        # 依次构建测量（测完释放，控制内存峰值）；顺序：dense → ivf → hnsw
        build_and_measure("dense（暴力）", make_dense, None, query_vecs,
                          questions, doc_ids, rand_queries, rows, scale)
        build_and_measure(f"ivf（nprobe={NPROBE}）", make_ivf, None, query_vecs,
                          questions, doc_ids, rand_queries, rows, scale)
        build_and_measure(f"hnsw（ef={EF_SEARCH}）", make_hnsw, None, query_vecs,
                          questions, doc_ids, rand_queries, rows, scale)
        del matrix, doc_ids
        print()

    # 汇总表
    print("=" * 88)
    print(f"{'规模(块)':>10}{'方案':<20}{'建库秒':>10}{'延迟ms':>10}{'加速比':>8}"
          f"{'Recall@10':>11}{'Recall@50':>11}")
    print("=" * 88)
    by_scale = {}
    for r in rows:
        by_scale.setdefault(r["规模(块)"], []).append(r)
    for scale in SCALE_LIST:
        group = by_scale[scale]
        dense_lat = next(r["平均延迟ms"] for r in group if r["方案"].startswith("dense"))
        for r in group:
            speedup = dense_lat / r["平均延迟ms"]
            print(f"{r['规模(块)']:>10,}{r['方案']:<20}{r['建库秒']:>10.2f}"
                  f"{r['平均延迟ms']:>10.3f}{speedup:>7.2f}x"
                  f"{r['Recall@10']:>10.2f}%{r['Recall@50']:>10.2f}%")
        print(f"{'':>10}{'向量化成本参考线':<20}{'':>10}{VECTORIZE_COST_MS:>10.0f}")
        print()

    csv_path = os.path.join(RESULTS_DIR, "scale_scan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"数据已保存到 {csv_path}")

    # 画规模-延迟双对数曲线（matplotlib 可用时）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors = {"dense": "#c0392b", "ivf": "#2980b9", "hnsw": "#27ae60"}
        for kind, label in (("dense", "dense 暴力检索"), ("ivf", "ivf nprobe=32"),
                            ("hnsw", "hnsw ef_search=64")):
            pts = [(r["规模(块)"], r["平均延迟ms"]) for r in rows if r["方案"].startswith(kind)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "o-", color=colors[kind], label=label)
            for x, y in pts:
                ax.annotate(f"{y:.1f}", (x, y), fontsize=8, textcoords="offset points",
                            xytext=(4, 4))
        ax.axhline(VECTORIZE_COST_MS, color="gray", linestyle="--", linewidth=1)
        ax.text(SCALE_LIST[0] * 1.05, VECTORIZE_COST_MS * 1.15,
                f"端到端向量化成本约 {VECTORIZE_COST_MS} ms（bge-m3）", fontsize=8, color="gray")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("语料规模（块数，对数轴）")
        ax.set_ylabel("纯检索延迟（毫秒，对数轴，不含向量化）")
        ax.set_title("规模扫描：暴力检索的线性增长与近似索引的平坦曲线")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        png_path = os.path.join(RESULTS_DIR, "scale_scan.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"曲线图已保存到 {png_path}")
    except ImportError:
        print("（未安装 matplotlib，跳过画图；数据见 CSV，可后续补图）")


if __name__ == "__main__":
    main()
