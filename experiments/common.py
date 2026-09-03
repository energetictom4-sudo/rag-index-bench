# -*- coding: utf-8 -*-
"""
实验公共模块：数据加载、向量化、文档级评测指标（BEIR 官方口径）

本模块只被 experiments/ 目录下的实验脚本引用，不修改也不依赖主程序代码；
仅通过 import rag_demo 复用主程序的嵌入客户端配置（Ollama + bge-m3），
保证实验与主程序使用完全一致的嵌入模型与接口。

口径说明（对齐 BEIR 官方评测）：
  - 索引单位：整篇文档（而非块）
  - 检索返回：Top-K 个文档
  - 评测分母：全量原始 qrels（不截断、不采样），score=1/2 作为相关度等级
  - 指标：文档级 Recall@K（分母=该题全部相关文档）与 NDCG@10（折损 1/log2(rank+1)）
"""

import json
import os
import sys

import faiss
import numpy as np

# 解决 Windows 控制台中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

# ================ 路径配置区（按本机实际情况调整） ================

# 项目目录（experiments/ 的上一级，即 D:/file/RAG/项目）
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据根目录：项目目录的上一级，存放 beir_data、beir_chunks.json、vector_store
DATA_ROOT = os.path.dirname(PROJECT_DIR)

DATASET = "nfcorpus"
CORPUS_FILE = os.path.join(DATA_ROOT, "beir_data", DATASET, "corpus.jsonl")   # 语料（每行一篇文档）
QUERIES_FILE = os.path.join(DATA_ROOT, "beir_data", DATASET, "queries.jsonl") # 测试查询
QRELS_FILE = os.path.join(DATA_ROOT, "beir_data", DATASET, "qrels", "test.tsv") # 全量相关标注
CHUNKS_FILE = os.path.join(DATA_ROOT, "beir_chunks.json")     # prepare_beir.py 生成的块文件（含doc_id）
BLOCK_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index.faiss")  # 现有块向量索引

# 实验产物目录（都在 experiments/ 下，不碰主程序任何文件）
CACHE_DIR = os.path.join(PROJECT_DIR, "experiments", "cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")
REPORTS_DIR = os.path.join(PROJECT_DIR, "experiments", "reports")

# ================ 评测参数 ================

RETRIEVAL_K = 100        # 检索返回文档数（必须 ≥ K_LIST 最大档位，满足 Recall@100）
K_LIST = (1, 5, 10, 20, 50, 100)   # Recall 统计档位
NDCG_K = 10              # NDCG 统计深度（BEIR 论文核心指标）
EMBED_BATCH = 256        # 向量化批次大小（与 indexes/dense.py 一致）

# BEIR 论文官方 nfcorpus NDCG@10 基线（用于直接对照）
BASELINES = [
    ("BGE-large-v1.5", 0.380),
    ("SPLADE++", 0.347),
    ("BM25 (Lucene)", 0.325),
    ("TAS-B / GenQ", 0.319),
    ("SBERT (msmarco)", 0.272),
]

# ================ 复用主程序的嵌入客户端（不动主程序，仅 import 读配置） ================

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
import rag_demo   # noqa: E402  复用 INDEX_CONFIG 中的 ollama_client / embedding_model

OLLAMA_CLIENT = rag_demo.INDEX_CONFIG["ollama_client"]
EMBEDDING_MODEL = rag_demo.INDEX_CONFIG["embedding_model"]


# ================ 数据加载 ================

def load_jsonl(path):
    """按行读取 jsonl 文件，返回列表"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_corpus_fulltexts():
    """从 corpus.jsonl 重建每篇文档的全文（title＋text），拼接方式与 prepare_beir.py 一致

    直接用语料原文而非块拼接：块之间有 50 字符重叠，拼接会产生重复文本。
    返回 (doc_ids, full_texts)，空文档按 prepare_beir.py 的规则跳过。
    """
    doc_ids, texts = [], []
    for doc in load_jsonl(CORPUS_FILE):
        doc_id = doc["_id"]
        text = (doc.get("text") or "").strip()
        title = (doc.get("title") or "").strip()
        full = f"{title}\n{text}" if title else text
        if not full:
            continue
        doc_ids.append(doc_id)
        texts.append(full)
    return doc_ids, texts


def load_queries():
    """读取全部测试查询，返回 {query_id: 查询文本}"""
    return {q["_id"]: q["text"] for q in load_jsonl(QUERIES_FILE)}


def load_qrels():
    """读取全量原始 qrels（不截断），返回 {query_id: {doc_id: score}}

    score 保留 1/2 的原始等级：NDCG 用等级加权，Recall 用 score>0 即相关。
    兼容三列与四列（TREC）两种格式，与 prepare_beir.py 的解析逻辑一致。
    """
    qrels = {}
    with open(QRELS_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts or parts[0] in ("query-id", "query_id", "qid"):
                continue
            if len(parts) == 4:
                qid, doc_id, score = parts[0], parts[2], parts[3]
            elif len(parts) == 3:
                qid, doc_id, score = parts[0], parts[1], parts[2]
            else:
                continue
            if int(score) > 0:   # 相关度>0 才计入相关（BEIR 官方口径）
                qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


# ================ 向量化（带断点续跑缓存，失败自动重试） ================

def embed_batch(texts, label=""):
    """分批向量化文本（OpenAI 兼容接口），返回 float32 矩阵 (len(texts), dim)

    单批失败自动重试一次（Ollama 偶发超时）；label 用于进度打印。
    """
    all_vecs = []
    n = len(texts)
    for i in range(0, n, EMBED_BATCH):
        end = min(i + EMBED_BATCH, n)
        batch = texts[i:end]
        response = None
        for attempt in range(2):   # 失败重试一次
            try:
                response = OLLAMA_CLIENT.embeddings.create(
                    model=EMBEDDING_MODEL, input=batch)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"  [向量化] 第{i}~{end}批失败（{e}），重试中...")
                else:
                    raise
        all_vecs.extend(item.embedding for item in response.data)
        print(f"  [向量化{label}] 进度：{end}/{n}")
    return np.array(all_vecs, dtype=np.float32)


def cached_embed(texts, ids, cache_prefix, label):
    """断点续跑的向量化：已缓存的 id 直接读回，未缓存的补向量化后落盘

    texts/ids 一一对应；缓存为 npy（完整矩阵）+ 文本文件（每行一个 id）。
    每批向量化完成后立即覆盖写盘完整矩阵，中断后重跑可续跑、不重复计算。
    返回与传入 texts 顺序一致的向量矩阵。
    """
    vec_path = os.path.join(CACHE_DIR, f"{cache_prefix}_vectors.npy")
    id_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.txt")

    # 1. 读取已有缓存（文件缺失或长度对不上则视为无缓存）
    cached_vecs, cached_ids = None, []
    if os.path.exists(vec_path) and os.path.exists(id_path):
        old_vecs = np.load(vec_path)
        with open(id_path, encoding="utf-8") as f:
            old_ids = [line.strip() for line in f if line.strip()]
        if len(old_vecs) == len(old_ids):
            cached_vecs, cached_ids = old_vecs, old_ids
            print(f"  [缓存] {label} 已有 {len(cached_ids)} 条，检查待补...")
        else:
            print(f"  [缓存] {cache_prefix} 缓存长度不一致，忽略缓存重新向量化")

    # 2. 找出未缓存的 id 并向量化（分批进行，每批完成后立即覆盖写盘实现断点续跑）
    cached_set = set(cached_ids)
    todo_idx = [i for i, did in enumerate(ids) if did not in cached_set]
    if todo_idx:
        new_vecs = embed_batch([texts[i] for i in todo_idx], label)
        merged_vecs = new_vecs if cached_vecs is None else np.vstack([cached_vecs, new_vecs])
        merged_ids = cached_ids + [ids[i] for i in todo_idx]
        np.save(vec_path, merged_vecs)
        with open(id_path, "w", encoding="utf-8") as f:
            f.write("\n".join(merged_ids) + "\n")
        print(f"  [缓存] {label} 本次新增 {len(todo_idx)} 条，累计 {len(merged_ids)} 条已落盘")
    else:
        merged_vecs, merged_ids = cached_vecs, cached_ids
        print(f"  [缓存] {label} 全部命中缓存（{len(ids)} 条），无需重新向量化")

    # 3. 按传入 ids 的顺序组装返回
    vec_map = dict(zip(merged_ids, merged_vecs))
    return np.vstack([vec_map[did] for did in ids])


# ================ FAISS 索引与检索 ================

def build_doc_index(doc_ids, doc_vectors):
    """构建文档级 FAISS 索引：向量 L2 归一化后 IndexFlatIP，内积即余弦相似度

    与 indexes/dense.py 的建库方式完全一致（仅索引单位从块换成文档）。
    返回 (index, doc_ids 列表)。
    """
    matrix = np.asarray(doc_vectors, dtype=np.float32).copy()
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, list(doc_ids)


def search_top_k(index, query_vectors, k):
    """批量检索：query_vectors 已归一化，返回每行查询对应的 Top-K 下标数组"""
    scores, ids = index.search(query_vectors, k)
    return ids, scores


# ================ 文档级评测指标（BEIR 官方口径） ================

def ndcg_at_k(ranked_scores, k):
    """单题 NDCG@k：ranked_scores 按检索顺序给出每个位置的文档相关度（0=不相关）

    BEIR 官方口径：收益=相关度等级（nfcorpus 为 1/2），折损=1/log2(rank+1)，rank 从 1 起。
    理想排序为该题全部相关度降序（只看前 k 位）。
    """
    top = ranked_scores[:k]
    dcg = sum(gain / np.log2(rank + 1) for rank, gain in enumerate(top, start=1))
    ideal = sorted(ranked_scores, reverse=True)[:k]
    idcg = sum(gain / np.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(ranked_doc_ids, qrels_for_q, k_list=K_LIST, ndcg_k=NDCG_K):
    """单题评测：给定检索返回的文档 id 序列与该题 qrels（{doc_id: score}）

    返回 dict：各档 Recall、NDCG@k、命中的相关文档 id 及排名、相关文档总数。
    召回分母 = 该题全量相关文档数，不截断（官方口径）。
    """
    rel_total = len(qrels_for_q)
    hit_ranks = []   # 命中的相关文档在结果中的排名（1 起）
    ranked_scores = []
    for rank, did in enumerate(ranked_doc_ids, start=1):
        score = qrels_for_q.get(did, 0)
        ranked_scores.append(score)
        if score > 0:
            hit_ranks.append(rank)

    recall = {}
    for k in k_list:
        hit_k = sum(1 for r in hit_ranks if r <= k)
        recall[f"recall@{k}"] = hit_k / rel_total if rel_total else 0.0

    return {
        "recall": recall,                       # 各档召回率
        "ndcg": ndcg_at_k(ranked_scores, ndcg_k),  # NDCG@k
        "rel_total": rel_total,                 # 该题全量相关文档数
        "hit_docs": [(d, r) for d, r in zip(ranked_doc_ids, ranked_scores)
                     if r > 0],                 # (命中的相关文档id, 排名)
        "hit_ranks": hit_ranks,
    }


def summarize(per_query_results, title, k_list=K_LIST, ndcg_k=NDCG_K):
    """汇总多题评测结果，返回格式化的多行文本（平均 Recall@K、平均 NDCG、零召回题数）"""
    lines = [title]
    n = len(per_query_results)
    if n == 0:
        lines.append("  （无数据）")
        return lines
    for k in k_list:
        avg = sum(r["recall"][f"recall@{k}"] for r in per_query_results) / n
        lines.append(f"  Recall@{k:<4} 平均 = {avg:.4f}")
    avg_ndcg = sum(r["ndcg"] for r in per_query_results) / n
    lines.append(f"  NDCG@{ndcg_k}   平均 = {avg_ndcg:.4f}")
    zero_recall = sum(1 for r in per_query_results if not r["hit_docs"])
    lines.append(f"  零召回题数（Recall@{max(k_list)}=0）= {zero_recall} / {n}")
    return lines


def baseline_comparison(avg_ndcg):
    """按实验方案预期的判断区间给出对标结论文字

    预期：bge-m3 作为强嵌入模型，合理落在 0.27~0.33；
    ≥ BM25 0.325 → 裸模板（无指令前缀）不是大问题；< SBERT 0.272 → 嵌入配置存在实质损伤。
    """
    lines = []
    lines.append("BEIR 论文官方 nfcorpus NDCG@10 基线：")
    for name, v in BASELINES:
        mark = "  ← 本实验" if abs(avg_ndcg - v) < 1e-9 else ""
        lines.append(f"    {name:<16} {v:.3f}{mark}")
    lines.append(f"    本实验方案 NDCG@10 = {avg_ndcg:.4f}")
    lines.append("")
    lines.append("对标判断：")
    if avg_ndcg >= 0.325:
        lines.append(f"  √ 达到/超过 BM25 基线（0.325）：裸模板（无指令前缀）不是大问题")
    elif avg_ndcg >= 0.272:
        lines.append(f"  √ 达到 SBERT 基线（0.272）以上：嵌入配置无实质损伤；")
        lines.append(f"    但与 BM25（0.325）仍有差距，可考虑查询指令前缀等进一步优化")
    else:
        lines.append(f"  × 低于 SBERT 基线（0.272）：坐实嵌入配置存在实质损伤，需排查")
    if 0.27 <= avg_ndcg <= 0.33:
        lines.append(f"  （数值落在预期区间 0.27~0.33 内，符合 bge-m3 裸模板的合理水平）")
    elif avg_ndcg > 0.33:
        lines.append(f"  （数值高于预期区间上限 0.33，表现超出预期）")
    else:
        lines.append(f"  （数值低于预期区间下限 0.27，偏弱，建议排查嵌入管线）")
    return lines
