# -*- coding: utf-8 -*-
"""
实验第 2 步：方案 B —— 块向量平均池化构造文档向量（零向量化成本）

思路（详见《文档级检索实验方案.docx》）：
  复用现有 vector_store/faiss_dense/ 里的块向量（17071 个，bge-m3 已归一化），
  按 doc_id 分组求平均、再归一化，作为每篇文档的向量，用同样的官方口径评测。
  A/B 对比结论很有价值：
    - 若 B≈A → 池化可行（省去 3633 次重新向量化，直接复用块向量）；
    - 若 A 明显更好 → 说明块级平均损失了文档的全局语义，需要重新向量化。

顺带产出"块级检索对照"：用原块向量索引检索 Top-50 块、映射回文档，
与全量 qrels 比对（即"现在的块级口径"对应的文档级数字），
供第 3 步对比"文档级 vs 块级"的差异。

本脚本不修改任何现有代码，只读现有块向量与块文件。

用法：
  python experiments/doc_retrieval_pool.py

输出：
  results/pool_doc_per_query.json        方案 B 逐题评测明细
  results/block_level_per_query.json     块级检索对照逐题明细
  reports/方案B_块向量池化_报告.txt         汇总报告
"""

import json
import os
import sys
import time

import faiss
import numpy as np

from common import (BLOCK_INDEX_FILE, CHUNKS_FILE, K_LIST, NDCG_K, REPORTS_DIR,
                    RESULTS_DIR, RETRIEVAL_K, baseline_comparison, build_doc_index,
                    cached_embed, evaluate_query, load_queries, load_qrels,
                    search_top_k, summarize)

RESULT_FILE = os.path.join(RESULTS_DIR, "pool_doc_per_query.json")
BLOCK_LEVEL_FILE = os.path.join(RESULTS_DIR, "block_level_per_query.json")
REPORT_FILE = os.path.join(REPORTS_DIR, "方案B_块向量池化_报告.txt")

# 块级对照的检索深度（与 eval_retrieval.py 的 QUERY_TOP_K=50 一致）
BLOCK_TOP_K = 50


def load_block_vectors():
    """读取现有块向量索引与块文件，返回 (块向量矩阵, 每块来源文档id列表)

    校验块文件与索引是否同一次构建（文本逐一相等），防止"张冠李戴"式的池化。
    """
    print("正在读取现有块向量索引...")
    index = faiss.read_index(BLOCK_INDEX_FILE)
    vectors = index.reconstruct_n(0, index.ntotal)
    print(f"  块向量索引：{index.ntotal} 个向量 × {index.d} 维（来自 {BLOCK_INDEX_FILE}）")

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunk_items = json.load(f)
    print(f"  块文件：{len(chunk_items)} 个块（来自 {CHUNKS_FILE}）")
    if len(chunk_items) != index.ntotal:
        raise RuntimeError(
            f"块文件（{len(chunk_items)} 个）与块向量索引（{index.ntotal} 个）数量不一致：\n"
            f"说明块文件与向量库不是同一次构建的产物，池化会张冠李戴。\n"
            f"请重新运行 prepare_beir.py 生成块文件并用主程序重新建库后再跑本实验。")

    # 文本逐一校验：块文件与索引内部 texts.json 的顺序是否一致
    texts_path = os.path.join(os.path.dirname(BLOCK_INDEX_FILE), "texts.json")
    with open(texts_path, encoding="utf-8") as f:
        index_texts = json.load(f)
    n_mismatch = sum(1 for a, b in zip(chunk_items, index_texts) if a["text"] != b)
    if n_mismatch:
        raise RuntimeError(
            f"块文件与向量库文本有 {n_mismatch} 处不一致，两者不是同一次构建，请重新生成。")
    print("  一致性校验通过：块文件与向量库文本逐一相同，doc_id 映射可靠")

    doc_ids = [item["doc_id"] for item in chunk_items]
    return vectors, doc_ids


def pool_by_doc(block_vectors, chunk_doc_ids):
    """按 doc_id 把块向量平均池化成文档向量（池化后再 L2 归一化）

    返回 (文档id列表, 文档向量矩阵)。块向量本身已归一化，平均后需重新归一化，
    检索内积才是余弦相似度。

    注意：！！！局限性：重要段落和次要段落被同等对待，可能损失关键信息。可能可以通过其他池化方法改进（可能为下一步优化方向）
    """
    unique_docs = sorted(set(chunk_doc_ids))
    doc_matrix = np.zeros((len(unique_docs), block_vectors.shape[1]), dtype=np.float32)
    for i, did in enumerate(unique_docs):
        mask = [d == did for d in chunk_doc_ids]
        doc_matrix[i] = block_vectors[mask].mean(axis=0)
    faiss.normalize_L2(doc_matrix)
    # 统计池化信息：每篇文档平均块数
    counts = np.array([sum(1 for d in chunk_doc_ids if d == did) for did in unique_docs])
    print(f"  池化完成：{len(unique_docs)} 篇文档（每篇平均 {counts.mean():.1f} 块，"
          f"最多 {counts.max()} 块）")
    return unique_docs, doc_matrix


def run_eval(index, index_ids, queries, qrels, qid_list, label, retr_k=RETRIEVAL_K,
             result_file=None):
    """共用评测流程：逐题检索、算指标、保存逐题结果，返回逐题结果列表"""
    query_texts = [queries[qid] for qid in qid_list]
    query_vectors = cached_embed(query_texts, qid_list, "query", "查询")
    per_query = []
    t_search_total = 0.0
    for qi, qid in enumerate(qid_list):
        qv = query_vectors[qi:qi + 1]
        faiss.normalize_L2(qv)
        t0 = time.perf_counter()
        ids_row, _ = search_top_k(index, qv, retr_k)
        t_search_total += time.perf_counter() - t0
        ranked_ids = [index_ids[i] for i in ids_row[0]]
        metrics = evaluate_query(ranked_ids, qrels[qid])
        per_query.append({
            "qid": qid,
            "question": queries[qid],
            "ndcg": metrics["ndcg"],
            "recall": metrics["recall"],
            "rel_total": metrics["rel_total"],
            "hit_docs": metrics["hit_docs"],
            "hit_ranks": metrics["hit_ranks"],
            "top_docs": ranked_ids[:10],
        })
        if (qi + 1) % 50 == 0:
            print(f"  [{label}] 进度：{qi + 1}/{len(qid_list)}")
    avg_ms = t_search_total / len(qid_list) * 1000
    print(f"  [{label}] 单题平均检索耗时（FAISS 搜索，不含向量化）{avg_ms:.2f} 毫秒")
    if result_file:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(per_query, f, ensure_ascii=False, indent=1)
        print(f"  [{label}] 逐题结果已保存到 {result_file}")
    return per_query


def main():
    report = []

    def log(s=""):
        print(s, flush=True)
        report.append(s)

    log("=" * 70)
    log("方案 B：块向量平均池化（零向量化成本，文档级检索）")
    log("=" * 70)

    # 1. 读现有块向量（零向量化成本）
    block_vectors, chunk_doc_ids = load_block_vectors()

    # 2. 池化构建文档向量
    t0 = time.perf_counter()
    doc_ids, doc_vectors = pool_by_doc(block_vectors, chunk_doc_ids)
    pool_time = time.perf_counter() - t0
    index, index_ids = build_doc_index(doc_ids, doc_vectors)
    log(f"  池化耗时 {pool_time:.2f} 秒，文档索引 {index.ntotal} 个向量")

    # 3. 加载全量查询与全量 qrels
    queries = load_queries()
    qrels = load_qrels()
    qid_list = [qid for qid in queries if qid in qrels]
    log(f"测试查询 {len(queries)} 条（有标注 {len(qid_list)} 条），全量 qrels 不截断")

    # 4. 方案 B 评测（文档级，官方口径）
    log(f"\n[1/2] 方案 B 文档级检索评测（Top-{RETRIEVAL_K}）...")
    pool_results = run_eval(index, index_ids, queries, qrels, qid_list,
                            "方案B", result_file=RESULT_FILE)

    log("")
    for line in summarize(pool_results, f"方案 B 汇总（{len(pool_results)} 题，全量 qrels 不截断）"):
        log(line)
    log("")
    for line in baseline_comparison(sum(r["ndcg"] for r in pool_results) / len(pool_results)):
        log(line)

    # 5. 块级检索对照（现在的块级口径：Top-50 块 → 映射文档 → 与全量 qrels 比对）
    log("")
    log("=" * 70)
    log(f"[2/2] 块级检索对照（现有口径：Top-{BLOCK_TOP_K} 块 → 映射文档）")
    log("=" * 70)
    block_matrix = np.asarray(block_vectors, dtype=np.float32).copy()
    block_index = faiss.IndexFlatIP(block_matrix.shape[1])
    block_index.add(block_matrix)   # 块向量已归一化，直接内积检索
    query_texts = [queries[qid] for qid in qid_list]
    query_vectors = cached_embed(query_texts, qid_list, "query", "查询")
    block_results = []
    for qi, qid in enumerate(qid_list):
        qv = query_vectors[qi:qi + 1]
        faiss.normalize_L2(qv)
        ids_row, _ = search_top_k(block_index, qv, BLOCK_TOP_K)   # 统一用common的检索封装（返回ids）
        # 命中块 → 来源文档集合（去重），这是块级检索能达到的"文档覆盖"
        covered_docs = []
        for bi in ids_row[0]:
            d = chunk_doc_ids[bi]
            if d not in covered_docs:
                covered_docs.append(d)
        metrics = evaluate_query(covered_docs, qrels[qid])
        block_results.append({
            "qid": qid,
            "question": queries[qid],
            "ndcg": metrics["ndcg"],   # 块级映射文档的 NDCG 仅作参考（口径已混合，非官方）
            "recall": metrics["recall"],
            "rel_total": metrics["rel_total"],
            "hit_docs": metrics["hit_docs"],
            "hit_ranks": metrics["hit_ranks"],
            "top_docs": covered_docs[:10],
        })
    log("")
    log(f"说明：Top-{BLOCK_TOP_K} 块去重映射后平均只覆盖 "
        f"{np.mean([len(r['top_docs']) for r in block_results]):.1f} 篇文档左右；")
    log("此口径与 BEIR 官方不完全一致（检索单位仍是块），仅用于对比差异。")
    for line in summarize(block_results, f"块级检索对照汇总（{len(block_results)} 题）"):
        log(line)
    with open(BLOCK_LEVEL_FILE, "w", encoding="utf-8") as f:
        json.dump(block_results, f, ensure_ascii=False, indent=1)
    log(f"\n块级对照逐题结果已保存到 {BLOCK_LEVEL_FILE}")

    # 6. 与方案 A 的对比留给 analyze.py（需方案 A 也跑完）
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    log(f"\n报告已保存到 {REPORT_FILE}")
    log("\n下一步：运行 python experiments/analyze.py 生成 A/B 对比与错误分析")


if __name__ == "__main__":
    main()
