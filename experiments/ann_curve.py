# -*- coding: utf-8 -*-
"""
ANN 召回率-延迟曲线生成器（阶段三核心实验工具）

对每个索引档位（dense 暴力参照 + ivf 各 nprobe + hnsw 各 ef_search），同时测量两个维度：
  1. 召回率：119 道题的真实查询向量（Ollama 向量化一次后缓存，不重复向量化），
     Top-50 检索 → 文档级判分（与 eval_retrieval.py 官方口径完全一致）
  2. 纯检索延迟：随机查询向量逐个 search 计时（与 bench_retrieval.py 同口径，不含向量化）
输出：
  - 控制台汇总表（召回率 + 延迟 + 加速比）
  - experiments/results/ann_curve.csv（供画图/论文引用）
  - experiments/results/ann_curve.png（召回率-延迟曲线，matplotlib 可用时生成）

用法：
  python experiments/ann_curve.py
前置：faiss_dense/、faiss_ivf/、faiss_hnsw/ 索引均已建好，题库为官方口径（119 题）
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
IVF_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_ivf", "index.faiss")
HNSW_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_hnsw", "index.faiss")
CACHE_DIR = os.path.join(PROJECT_DIR, "experiments", "cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

TOPK = 50            # 检索返回片段数（与 eval_retrieval.py 的 QUERY_TOP_K 一致）
RECALL_KS = (5, 10, 20, 50)   # 召回率统计档位
NPROBE_LIST = (1, 4, 8, 16, 32, 64, 128, 256)   # ivf 扫描的 nprobe 档位
EF_SEARCH_LIST = (4, 8, 16, 32, 64, 128, 256)   # hnsw 扫描的 ef_search 档位（与 nprobe 对齐）
N_LATENCY_QUERIES = 1000   # 延迟测量的随机查询数
WARMUP = 10                # 延迟测量预热次数
SEED = 42                  # 随机查询向量种子

# ======================== 以下为代码逻辑，无需改动 ========================


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_query_vectors(questions: list) -> np.ndarray:
    """119 道题查询向量化（缓存到 cache/，只向量化一次）"""
    vec_path = os.path.join(CACHE_DIR, "ann_query_vecs.npy")
    if os.path.exists(vec_path):
        vecs = np.load(vec_path)
        if len(vecs) == len(questions):
            print(f"[查询向量] 命中缓存（{len(vecs)} 题），跳过向量化")
            return vecs
    # 复用主程序嵌入客户端（与评测完全一致的模型与接口）
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    import rag_demo   # noqa: E402
    client = rag_demo.INDEX_CONFIG["ollama_client"]
    model = rag_demo.INDEX_CONFIG["embedding_model"]
    texts = [q["问题"] for q in questions]
    print(f"[查询向量] 正在向量化 {len(texts)} 道题（Ollama bge-m3）...")
    response = client.embeddings.create(model=model, input=texts)
    vecs = np.array([item.embedding for item in response.data], dtype=np.float32)
    faiss.normalize_L2(vecs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(vec_path, vecs)
    return vecs


def evaluate_recall(index: faiss.Index, query_vecs: np.ndarray, questions: list,
                    chunk_doc_ids: list, topk: int) -> dict:
    """全部题的文档级召回率（官方口径，与 eval_retrieval.py 判分一致）

    注意：index.search() 返回 (分数, ids)，用 ids 判分；逐题 search，
    与 eval_retrieval.py 的调用方式保持一致。
    """
    per_k_hits = {k: [] for k in RECALL_KS}
    for qi, q in enumerate(questions):
        gold_docs = {chunk_doc_ids[i] for i in q["相关块编号"]}
        if not gold_docs:
            continue
        _, row = index.search(query_vecs[qi:qi + 1], topk)   # search 返回 (分数, ids)，丢弃分数
        covered = [chunk_doc_ids[int(i)] for i in row[0]]
        for k in RECALL_KS:
            hit = len(set(covered[:k]) & gold_docs)
            per_k_hits[k].append(hit / len(gold_docs))
    return {f"recall@{k}": float(np.mean(v)) for k, v in per_k_hits.items()}


def bench_latency(index: faiss.Index, queries: np.ndarray, topk: int) -> dict:
    """纯检索延迟（与 bench_retrieval.py 同口径），返回毫秒统计"""
    for q in queries[:WARMUP]:
        index.search(q.reshape(1, -1), topk)
    times = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), topk)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {"avg_ms": float(arr.mean()), "median_ms": float(np.median(arr))}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chunk_items = load_json(CHUNKS_FILE)
    chunk_doc_ids = [c["doc_id"] for c in chunk_items]
    questions = load_json(QUESTIONS_FILE)
    print(f"数据：{len(chunk_items)} 个块，{len(questions)} 道题（官方口径）\n")

    query_vecs = load_query_vectors(questions)
    dense_index = faiss.read_index(DENSE_INDEX_FILE)
    ivf_index = faiss.read_index(IVF_INDEX_FILE)
    hnsw_index = faiss.read_index(HNSW_INDEX_FILE)
    rng = np.random.default_rng(SEED)
    rand_queries = rng.standard_normal((N_LATENCY_QUERIES, dense_index.d)).astype(np.float32)
    faiss.normalize_L2(rand_queries)

    # 档位列表：(名称, 索引, 参数种类, 参数值)；种类 None=nprobe=ef 对应三种检索方式
    runs = [("dense (IndexFlatIP)", dense_index, None, None)]
    runs += [(f"ivf (nprobe={nprobe})", ivf_index, "nprobe", nprobe) for nprobe in NPROBE_LIST]
    runs += [(f"hnsw (ef={ef})", hnsw_index, "ef", ef) for ef in EF_SEARCH_LIST]

    rows = []
    print("=" * 78)
    print(f"{'档位':<22}{'Recall@5':>9}{'Recall@10':>10}{'Recall@20':>10}{'Recall@50':>10}"
          f"{'延迟ms':>9}{'加速比':>8}")
    print("=" * 78)
    dense_avg_ms = None
    for name, index, kind, value in runs:
        if kind == "nprobe":
            index.nprobe = value
        elif kind == "ef":
            index.hnsw.efSearch = value
        rec = evaluate_recall(index, query_vecs, questions, chunk_doc_ids, TOPK)
        lat = bench_latency(index, rand_queries, TOPK)
        if dense_avg_ms is None:
            dense_avg_ms = lat["avg_ms"]
        speedup = dense_avg_ms / lat["avg_ms"]
        row = {"档位": name, **rec, "平均延迟ms": round(lat["avg_ms"], 3),
               "中位延迟ms": round(lat["median_ms"], 3), "加速比": round(speedup, 2)}
        rows.append(row)
        print(f"{name:<22}{rec['recall@5']:>9.2%}{rec['recall@10']:>10.2%}"
              f"{rec['recall@20']:>10.2%}{rec['recall@50']:>10.2%}"
              f"{lat['avg_ms']:>9.3f}{speedup:>7.2f}x")

    # 保存 CSV（供画图与论文引用）
    csv_path = os.path.join(RESULTS_DIR, "ann_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n数据已保存到 {csv_path}")

    # 画召回率-延迟曲线（matplotlib 可用时）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]   # 中文字体
        plt.rcParams["axes.unicode_minus"] = False                        # 负号正常显示
        xs = [r["平均延迟ms"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, style in (("recall@10", "o-"), ("recall@50", "s--")):
            ys = [r[k] * 100 for r in rows]
            ax.plot(xs, ys, style, label=f"{k} 召回率")
        ax.set_xscale("log")
        ax.set_xlabel("纯检索延迟（毫秒，对数轴，不含向量化）")
        ax.set_ylabel("召回率（%）")
        ax.set_title("暴力检索 vs IVF(nprobe) vs HNSW(ef_search)：召回率-延迟曲线（nfcorpus 17071 块）")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        for i, r in enumerate(rows):
            ax.annotate(r["档位"].split("(")[-1].rstrip(")"),
                        (xs[i], rows[i]["recall@10"] * 100), fontsize=8)
        png_path = os.path.join(RESULTS_DIR, "ann_curve.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"曲线图已保存到 {png_path}")
    except ImportError:
        print("（未安装 matplotlib，跳过画图；数据见 CSV，可后续补图）")


if __name__ == "__main__":
    main()
