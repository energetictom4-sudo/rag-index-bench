# -*- coding: utf-8 -*-
"""IVF 的 k-means 种子稳定性检验（批判点修正：统计严谨性）

问题：主表所有数字来自单一 k-means 初始化（默认 seed），簇划分不同时 IVF 的
召回会变多少？本脚本用 3 个不同种子重新训练 k-means、重建 IndexIVFFlat，
按与 eval_retrieval.py 完全相同的块级口径评测，报告均值±标准差。

用法：
  python experiments/ivf_seed_stability.py

前提：vector_store/faiss_dense 为当前数据集（nfcorpus）的块向量，题库为 beir_题库.json。
输出：experiments/results/ivf_seed_stability.csv
"""

import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cached_embed  # noqa: E402  复用查询向量缓存

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)
BLOCK_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index.faiss")
QUESTIONS_FILE = os.path.join(DATA_ROOT, "beir_题库.json")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

SEEDS = (1, 2, 3)
NLIST = 256
NPROBE = 32
TOP_K = 50


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def block_level_recall(ivf, queries, qids, questions, chunk_doc_ids, k=TOP_K):
    """文档级召回：与 eval_retrieval.py 的 Recall@50 口径一致（每题检索 Top-50 块，
    命中块映射到来源文档后与"相关文档集合"比对；召回 = 命中相关文档数 / 相关文档总数）"""
    qv = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    _, ids = ivf.search(qv.astype(np.float32), k)   # faiss 返回 (scores, ids)
    recalls = []
    for i, qid in enumerate(qids):
        gold_docs = {chunk_doc_ids[b] for b in questions[qid] if 0 <= b < len(chunk_doc_ids)}
        covered_docs = {chunk_doc_ids[b] for b in ids[i].tolist() if 0 <= b < len(chunk_doc_ids)}
        hit = len(gold_docs & covered_docs)
        recalls.append(hit / len(gold_docs) if gold_docs else 0.0)
    return recalls


def main():
    print("=" * 70)
    print(f"IVF k-means 种子稳定性（seeds={SEEDS}，nlist={NLIST}/nprobe={NPROBE}）")
    print("=" * 70)

    block_index = faiss.read_index(BLOCK_INDEX_FILE)
    vecs = block_index.reconstruct_n(0, block_index.ntotal)
    d = vecs.shape[1]
    print(f"块向量：{block_index.ntotal} × {d}")

    questions = load_questions()
    qid_list = [q["id"] for q in questions]
    query_texts = [q["问题"] for q in questions]
    query_vecs = cached_embed(query_texts, qid_list, "query_nfcorpus_seeds", "查询")
    q_by_id = {q["id"]: set(q["相关块编号"]) for q in questions}
    # 块编号 → 来源文档 id（块文件与块向量同源，顺序一致）
    with open(os.path.join(DATA_ROOT, "beir_chunks.json"), encoding="utf-8") as f:
        chunk_doc_ids = [it["doc_id"] for it in json.load(f)]
    print(f"题库：{len(questions)} 题（文档级 Recall@50 口径，与 eval_retrieval.py 一致）")

    print("\n各种子块级 Recall@50：")
    rows = [["种子", "Recall@50"]]
    recalls_all = []
    for s in SEEDS:
        kmeans = faiss.Kmeans(d, NLIST, niter=25, seed=s, verbose=False)
        kmeans.train(vecs)
        centroids = np.ascontiguousarray(kmeans.centroids)
        faiss.normalize_L2(centroids)
        quantizer = faiss.IndexFlatIP(d)
        quantizer.add(centroids)
        ivf = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)
        ivf.is_trained = True   # 质心已由外部 k-means 装好，跳过内部聚类训练
        ivf.nprobe = NPROBE
        ivf.add(vecs)
        recalls = block_level_recall(ivf, query_vecs, qid_list, q_by_id, chunk_doc_ids)
        avg = sum(recalls) / len(recalls)
        recalls_all.append(avg)
        print(f"  seed={s}: Recall@50 = {avg:.2%}")
        rows.append([s, round(avg, 4)])

    mean, std = np.mean(recalls_all), np.std(recalls_all)
    print(f"\n均值 {mean:.2%} ± {std:.2%}（3 种子）")
    print(f"对照：主表 ivf 行 Recall@50 = 48.68%（faiss 默认 seed 构建）")

    csv_path = os.path.join(RESULTS_DIR, "ivf_seed_stability.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["种子", "Recall@50"])
        for r in rows[1:]:
            w.writerow(r)
        w.writerow([])
        w.writerow(["均值", round(mean, 4)])
        w.writerow(["标准差", round(std, 4)])
    print(f"已写入 {csv_path}")


if __name__ == "__main__":
    main()
