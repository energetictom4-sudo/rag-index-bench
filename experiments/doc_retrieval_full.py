# -*- coding: utf-8 -*-
"""
实验第 1 步：方案 A —— 整篇文档直接向量化（最"官方"的做法）

背景（详见《文档级检索实验方案.docx》）：
  现有主程序是块级检索（17071 个块），与 BEIR 官方评测口径（文档级）不一致，
  导致评测结果无法直接对照 BEIR 论文官方基线。本脚本把索引单位从块换成整篇文档：
    - 直接从 corpus.jsonl 重建每篇文档全文（title＋text，与 prepare_beir.py 一致），
      不用块拼接（块之间有 50 字符重叠，拼接会产生重复文本）；
    - 3633 篇文档向量化（块级建库是 17071 次请求的约 1/4.7），几分钟跑完；
    - 用全量原始 qrels（12334 条，不截断）算文档级 Recall@K 与 NDCG@10（官方口径）。

本脚本不修改任何现有代码（rag_demo.py / indexes/dense.py / eval_retrieval.py 均不动），
产物全部放在 experiments/ 目录下（向量缓存 cache/、逐题结果 results/、报告 reports/）。

用法：
  python experiments/doc_retrieval_full.py            # 完整跑（断点续跑，中断重跑即可）
  python experiments/doc_retrieval_full.py --no-embed # 向量已缓存时跳过向量化（调试用）

输出：
  results/full_doc_per_query.json   逐题评测明细（供 analyze.py 错误分析用）
  reports/方案A_整文档向量化_报告.txt  汇总报告
"""

import json
import os
import sys
import time

import faiss
import numpy as np

from common import (BASELINES, CACHE_DIR, K_LIST, NDCG_K, QUERIES_FILE, QRELS_FILE,
                    REPORTS_DIR, RESULTS_DIR, RETRIEVAL_K, baseline_comparison,
                    build_doc_index, cached_embed, evaluate_query, load_corpus_fulltexts,
                    load_queries, load_qrels, search_top_k, summarize)

RESULT_FILE = os.path.join(RESULTS_DIR, "full_doc_per_query.json")
REPORT_FILE = os.path.join(REPORTS_DIR, "方案A_整文档向量化_报告.txt")


def main():
    report = []   # 报告内容收集（控制台打印 + 写入 UTF-8 文件）

    def log(s=""):
        print(s, flush=True)
        report.append(s)

    no_embed = "--no-embed" in sys.argv

    log("=" * 70)
    log("方案 A：整篇文档直接向量化（文档级检索，BEIR 官方口径）")
    log("=" * 70)

    # 1. 从 corpus.jsonl 重建全文（title＋text，与 prepare_beir.py 拼接方式一致）
    doc_ids, full_texts = load_corpus_fulltexts()
    avg_len = np.mean([len(t) for t in full_texts])
    log(f"语料共 {len(doc_ids)} 篇文档，平均长度 {avg_len:.0f} 字符")

    # 2. 全文向量化（带断点续跑缓存：中断后重跑会接着上次进度继续）
    log(f"\n[1/4] 向量化 {len(full_texts)} 篇文档全文（bge-m3）...")
    t0 = time.perf_counter()
    if no_embed:
        log("  已指定 --no-embed，跳过向量化（要求缓存已完整）")
    doc_vectors = cached_embed(full_texts, doc_ids, "full_doc", "全文")
    embed_time = time.perf_counter() - t0
    log(f"  向量化耗时 {embed_time:.1f} 秒（含缓存命中部分）")

    # 3. 构建文档级 FAISS 索引（L2 归一化 + IndexFlatIP = 余弦相似度）
    log(f"\n[2/4] 构建文档级 FAISS 索引...")
    t0 = time.perf_counter()
    index, index_doc_ids = build_doc_index(doc_ids, doc_vectors)
    build_time = time.perf_counter() - t0
    log(f"  索引规模 {index.ntotal} 个向量 × {index.d} 维，构建耗时 {build_time:.2f} 秒")

    # 4. 加载全量查询与全量 qrels（不截断）
    queries = load_queries()
    qrels = load_qrels()
    qid_list = [qid for qid in queries if qid in qrels]
    n_qrel_pairs = sum(len(v) for v in qrels.values())
    log(f"\n[3/4] 测试查询 {len(queries)} 条（有标注 {len(qid_list)} 条），"
        f"全量 qrels 共 {n_qrel_pairs} 条")

    # 检查 qrels 中的文档是否都在索引里（缺失文档永远检索不到，报告里要说明）
    index_id_set = set(index_doc_ids)
    missing_pairs = sum(1 for v in qrels.values() for d in v if d not in index_id_set)
    if missing_pairs:
        log(f"  注意：有 {missing_pairs} 条 qrels 的文档不在索引中（语料为空被跳过），"
            f"其召回分母仍按全量 qrels 统计（官方口径）")

    # 5. 查询向量化（与方案 B 共用同一份缓存，保证 A/B 查询侧完全一致）
    log(f"\n[4/4] 逐题检索并评测（Top-{RETRIEVAL_K}，指标：Recall@K + NDCG@{NDCG_K}）...")
    query_texts = [queries[qid] for qid in qid_list]
    query_vectors = cached_embed(query_texts, qid_list, "query", "查询")

    per_query = []
    t_search_total = 0.0
    for qi, qid in enumerate(qid_list):
        qv = query_vectors[qi:qi + 1]
        faiss.normalize_L2(qv)
        t0 = time.perf_counter()
        ids_row, _ = search_top_k(index, qv, RETRIEVAL_K)
        t_search_total += time.perf_counter() - t0
        ranked_doc_ids = [index_doc_ids[i] for i in ids_row[0]]
        metrics = evaluate_query(ranked_doc_ids, qrels[qid])
        per_query.append({
            "qid": qid,
            "question": queries[qid],
            "ndcg": metrics["ndcg"],
            "recall": metrics["recall"],
            "rel_total": metrics["rel_total"],
            "hit_docs": metrics["hit_docs"],
            "hit_ranks": metrics["hit_ranks"],
            "top_docs": ranked_doc_ids[:10],   # 保留 Top-10 文档 id 供错误分析
        })
        if (qi + 1) % 50 == 0:
            log(f"  进度：{qi + 1}/{len(qid_list)}")

    avg_search_ms = t_search_total / len(qid_list) * 1000
    log(f"  检索完成，单题平均检索耗时（FAISS 搜索，不含向量化）{avg_search_ms:.2f} 毫秒")

    # 6. 保存逐题结果与汇总报告
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=1)
    log(f"\n逐题结果已保存到 {RESULT_FILE}")

    log("")
    for line in summarize(per_query, f"方案 A 汇总（{len(per_query)} 题，全量 qrels 不截断）"):
        log(line)
    log("")
    for line in baseline_comparison(sum(r["ndcg"] for r in per_query) / len(per_query)):
        log(line)

    # 与方案 B 的对比在 analyze.py 中统一生成（需两个方案都跑完）

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    log(f"\n报告已保存到 {REPORT_FILE}")


if __name__ == "__main__":
    main()
