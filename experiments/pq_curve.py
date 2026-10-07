# -*- coding: utf-8 -*-
"""
PQ 压缩比-召回-延迟曲线生成器（阶段三存储压缩实验工具）

固定 nprobe=32（与 ivf 甜蜜点一致），扫描 PQ 子段数 M（16/32/64），对每档同时测量：
  1. 压缩比：向量部分 float32 全量（4096 字节/块）→ M 字节/块，以及索引文件实际大小
  2. 召回率：119 道题的真实查询向量（复用 ann_curve.py 的缓存，不重复向量化），
     Top-50 检索 → 文档级判分（与 eval_retrieval.py 官方口径完全一致）
  3. 纯检索延迟：随机查询向量逐个 search 计时（与 bench_retrieval.py 同口径，不含向量化）
对比行：dense 暴力检索与 ivf nprobe=32（M=∞ 即无量化）。
输出：
  - 控制台汇总表（M、压缩比、索引文件大小、召回率、延迟、加速比）
  - experiments/results/pq_curve.csv（供画图/论文引用）

用法：
  python experiments/pq_curve.py
前置：faiss_dense/、faiss_ivf/ 索引均已建好，题库为官方口径（119 题）
"""

import csv
import json
import os
import sys
import tempfile
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
IVF_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_ivf", "index.faiss")
CACHE_DIR = os.path.join(PROJECT_DIR, "experiments", "cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

TOPK = 50            # 检索返回片段数（与 eval_retrieval.py 的 QUERY_TOP_K 一致）
RECALL_KS = (10, 50)          # 召回率统计档位（压缩实验关注中深档位）
NLIST = 256          # 聚类簇数（与 ivf/pq 方案一致）
NPROBE = 32          # 搜索簇数（与 ivf 甜蜜点一致）
N_BITS = 8           # 每子段码字位数（标准 8 位 = 256 码字）
M_LIST = (16, 32, 64)          # PQ 子段数扫描档位：越小压缩越狠、量化越粗
N_LATENCY_QUERIES = 1000   # 延迟测量的随机查询数
WARMUP = 10                # 延迟测量预热次数
SEED = 42                  # 随机查询向量种子

# ======================== 以下为代码逻辑，无需改动 ========================


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_recall(index: faiss.Index, query_vecs: np.ndarray, questions: list,
                    chunk_doc_ids: list, topk: int) -> dict:
    """全部题的文档级召回率（官方口径，与 ann_curve.py / eval_retrieval.py 判分一致）"""
    per_k_hits = {k: [] for k in RECALL_KS}
    for qi, q in enumerate(questions):
        gold_docs = {chunk_doc_ids[i] for i in q["相关块编号"]}
        if not gold_docs:
            continue
        _, row = index.search(query_vecs[qi:qi + 1], topk)
        covered = [chunk_doc_ids[int(i)] for i in row[0]]
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


def index_file_size_mb(index: faiss.Index) -> float:
    """索引落盘后的实际文件大小（写临时文件测量，不污染 vector_store）"""
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        tmp_path = f.name
    try:
        faiss.write_index(index, tmp_path)
        return os.path.getsize(tmp_path) / 1024 / 1024
    finally:
        os.remove(tmp_path)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chunk_items = load_json(CHUNKS_FILE)
    chunk_doc_ids = [c["doc_id"] for c in chunk_items]
    questions = load_json(QUESTIONS_FILE)
    print(f"数据：{len(chunk_items)} 个块，{len(questions)} 道题（官方口径），"
          f"固定 nprobe={NPROBE}、nlist={NLIST}\n")

    # 复用 ann_curve.py 缓存的查询向量（无缓存时走同一向量化逻辑）
    from ann_curve import load_query_vectors
    query_vecs = load_query_vectors(questions)

    dense_index = faiss.read_index(DENSE_INDEX_FILE)
    ivf_index = faiss.read_index(IVF_INDEX_FILE)
    ivf_index.nprobe = NPROBE
    d = dense_index.d
    matrix = np.array(dense_index.reconstruct_n(0, dense_index.ntotal), dtype=np.float32)
    raw_mb = matrix.nbytes / 1024 / 1024

    rng = np.random.default_rng(SEED)
    rand_queries = rng.standard_normal((N_LATENCY_QUERIES, d)).astype(np.float32)
    faiss.normalize_L2(rand_queries)

    # 参照行：dense 暴力（M=∞ 无量化无聚类）、ivf nprobe=32（M=∞ 无量化）
    rows = []
    for name, index in (("dense（暴力参照）", dense_index), ("ivf nprobe=32（无量化）", ivf_index)):
        rec = evaluate_recall(index, query_vecs, questions, chunk_doc_ids, TOPK)
        lat = bench_latency(index, rand_queries, TOPK)
        rows.append({
            "档位": name, "M": "∞", "压缩比": 1.0,
            "索引文件MB": round(index_file_size_mb(index), 2),
            "Recall@10": round(rec["recall@10"] * 100, 2),
            "Recall@50": round(rec["recall@50"] * 100, 2),
            "平均延迟ms": round(lat, 3),
        })

    # 扫描各 M 档位：内存建 IVFPQ 索引（train + add），测压缩/召回/延迟
    for m in M_LIST:
        print(f"[pq_curve] 构建 M={m}（每块 {d // m} 维 × {2 ** N_BITS} 码字，压缩 {4096 // m} 倍）...")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, NLIST, m, N_BITS, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.add(matrix)
        index.nprobe = NPROBE
        rec = evaluate_recall(index, query_vecs, questions, chunk_doc_ids, TOPK)
        lat = bench_latency(index, rand_queries, TOPK)
        rows.append({
            "档位": f"ivfpq M={m}（nprobe={NPROBE}）", "M": str(m),
            "压缩比": 4096 // m,
            "索引文件MB": round(index_file_size_mb(index), 2),
            "Recall@10": round(rec["recall@10"] * 100, 2),
            "Recall@50": round(rec["recall@50"] * 100, 2),
            "平均延迟ms": round(lat, 3),
        })
        print(f"  压缩 {4096 // m} 倍 → 索引 {rows[-1]['索引文件MB']} MB，"
              f"Recall@50={rows[-1]['Recall@50']}%，延迟 {rows[-1]['平均延迟ms']} ms")

    # 汇总表
    print()
    print("=" * 96)
    print(f"{'档位':<30}{'压缩比':>8}{'索引MB':>10}{'Recall@10':>11}{'Recall@50':>11}{'延迟ms':>10}")
    print("=" * 96)
    for r in rows:
        print(f"{r['档位']:<30}{r['压缩比']:>7}x{r['索引文件MB']:>10.2f}"
              f"{r['Recall@10']:>10.2f}%{r['Recall@50']:>10.2f}%{r['平均延迟ms']:>10.3f}")
    print(f"\n说明：向量部分原始大小 {raw_mb:.1f} MB（float32，4096 字节/块）；"
          f"压缩比 = 原始字节数 / PQ 码字节数。")

    csv_path = os.path.join(RESULTS_DIR, "pq_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"数据已保存到 {csv_path}")


if __name__ == "__main__":
    main()
