# -*- coding: utf-8 -*-
"""文档级官方口径评测（批判点修正：主表全部改用 BEIR 官方指标）

BEIR 官方评测口径：检索单位=整篇文档、qrels=全量不截断、核心指标=NDCG@10
（nfcorpus 分级相关度 1/2 参与 NDCG 加权；scifact/fiqa 二值）。本脚本把
8 个索引方案统一到这个口径下重跑，产出与块级主表并行的"官方口径对照表"，
并顺带完成 BM25 自实现与 Lucene 官方基线（nfcorpus NDCG@10=0.325）的对标。

各方案在文档级口径下的检索形态：
  dense        文档全文向量化（IndexFlatIP）
  doc_dense    块向量按文档平均池化（与 indexes/doc_dense.py 一致）
  parent_doc   块级 Top-200 → 映射文档去重（与 indexes/parent_doc.py 一致）
  bm25         文档级倒排（复用 indexes/bm25.py 的 tokenize 与 k1=1.5/b=0.75）
  hybrid       RRF 融合 doc_dense 与 bm25 的文档排名（k=60，与 indexes/hybrid.py 一致）
  ivf/hnsw/pq  复用对应块级近似索引 Top-200 → 映射文档（nprobe=32/ef_search=64）

用法：
  python experiments/eval_doc_level.py --method all --dataset nfcorpus
  python experiments/eval_doc_level.py --method bm25 --dataset scifact

输出：experiments/results/doc_level_{method}_{ds}.csv + 控制台汇总
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import faiss
import numpy as np
from nltk.stem import PorterStemmer   # 仅对标实验用（BEIR 官方 Lucene 分析链含 Porter 词干化）

# 复用 experiments/common.py 的评测函数与嵌入客户端（不依赖其路径常量）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cached_embed, ndcg_at_k, OLLAMA_CLIENT  # noqa: E402
from indexes.bm25 import tokenize, K1, B  # noqa: E402  复用分词与标准 BM25 参数

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

RETRIEVAL_K = 100          # 官方口径检索深度（支持 Recall@100）
BLOCK_TOP = 200            # 块级方案映射文档的块检索深度（与 parent_doc 的 SUB_TOP 一致）
NDCG_K = 10
K_LIST = (5, 10, 20, 50, 100)

# 近似方案参数（与 indexes/ 各方案一致）
NPROBE = 32
EF_SEARCH = 64

BASELINES_NFCORPUS = [("BGE-large-v1.5", 0.380), ("SPLADE++", 0.347),
                      ("BM25 (Lucene)", 0.325), ("TAS-B / GenQ", 0.319),
                      ("SBERT (msmarco)", 0.272)]


def ds_files(ds: str) -> dict:
    """当前数据集的各数据文件路径（命名规则与 eval_retrieval.py 一致）"""
    chunks = (os.path.join(DATA_ROOT, "beir_chunks.json") if ds == "nfcorpus"
              else os.path.join(DATA_ROOT, f"beir_chunks_{ds}.json"))
    return {
        "corpus": os.path.join(DATA_ROOT, "beir_data", ds, "corpus.jsonl"),
        "queries": os.path.join(DATA_ROOT, "beir_data", ds, "queries.jsonl"),
        "qrels": os.path.join(DATA_ROOT, "beir_data", ds, "qrels", "test.tsv"),
        "chunks": chunks,
        "block_index": os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index.faiss"),
    }


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_queries(path):
    return {q["_id"]: q["text"] for q in load_jsonl(path)}


def load_qrels(path):
    """全量 qrels，保留原始分级 score（nfcorpus 1/2 参与 NDCG 加权，官方口径）"""
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts or parts[0] in ("query-id", "query_id", "qid"):
                continue
            if len(parts) == 4:
                qid, did, score = parts[0], parts[2], parts[3]
            elif len(parts) == 3:
                qid, did, score = parts[0], parts[1], parts[2]
            else:
                continue
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_corpus(path, include_title=True):
    """语料全文（默认 title+text 拼接与 prepare_beir.py 一致，空文档跳过）

    include_title=False 用于对标实验：BEIR 官方 Lucene 基线的语料构造不含标题时，
    可定位"标题拼接"对 BM25 分数的影响。
    """
    docs = {}
    for doc in load_jsonl(path):
        text = (doc.get("text") or "").strip()
        title = (doc.get("title") or "").strip() if include_title else ""
        full = f"{title}\n{text}" if title else text
        if full:
            docs[doc["_id"]] = full
    return docs


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [it["text"] for it in items], [it["doc_id"] for it in items]


def load_block_vectors(index_path, chunk_texts, expect_n):
    """读块向量索引并校验与块文件同源"""
    index = faiss.read_index(index_path)
    if index.ntotal != expect_n:
        raise RuntimeError(f"块向量索引（{index.ntotal}）与块文件（{expect_n}）数量不一致："
                           f"当前 vector_store 索引不是 {os.path.basename(index_path)} 数据集的，"
                           f"请先用 eval_retrieval.py 跑一次 dense 建库")
    texts_path = os.path.join(os.path.dirname(index_path), "texts.json")
    with open(texts_path, encoding="utf-8") as f:
        index_texts = json.load(f)
    if any(a != b for a, b in zip(chunk_texts, index_texts)):
        raise RuntimeError("块文件与向量库文本不一致，非同一数据集构建")
    return index.reconstruct_n(0, index.ntotal)


# ---------- 各方法的文档级检索函数构造 ----------

def make_dense_searcher(doc_ids, full_texts, ds):
    vecs = cached_embed(full_texts, doc_ids, f"full_{ds}", "全文")
    index = faiss.IndexFlatIP(vecs.shape[1])
    faiss.normalize_L2(vecs)
    index.add(vecs)

    def search(qv, k):
        _, ids = index.search(qv, k)
        return [doc_ids[i] for i in ids[0]]
    return search


def make_pool_searcher(block_vecs, chunk_doc_ids):
    # 一趟分组累加 O(块数)：旧实现按"每文档扫描全部块"是 O(文档数×块数)（2026-10-15 修复）
    doc_ids = sorted(set(chunk_doc_ids))
    doc_index = {did: i for i, did in enumerate(doc_ids)}
    sums = np.zeros((len(doc_ids), block_vecs.shape[1]), dtype=np.float32)
    counts = np.zeros(len(doc_ids), dtype=np.int32)
    for j, did in enumerate(chunk_doc_ids):
        i = doc_index[did]
        sums[i] += block_vecs[j]
        counts[i] += 1
    # counts 转 float32 再除：float32/int32 在旧 numpy 会提升为 float64，faiss 拒绝
    matrix = (sums / counts.astype(np.float32)[:, None]).astype(np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    def search(qv, k):
        _, ids = index.search(qv, k)
        return [doc_ids[i] for i in ids[0]]
    return search


def make_block_to_doc_searcher(block_index, chunk_doc_ids):
    """块级检索（含近似索引）→ 文档去重映射，与 parent_doc/近似方案语义一致"""
    def search(qv, k):
        _, ids = block_index.search(qv, BLOCK_TOP)
        docs = []
        for bi in ids[0]:
            d = chunk_doc_ids[bi]
            if d not in docs:
                docs.append(d)
        return docs[:k]
    return search


def make_bm25_doc_searcher(doc_ids, full_texts, k1=K1, b=B, stem=False):
    """文档级 BM25 倒排（复用 indexes/bm25.py 的分词；k1/b 可切换以对标：
    本项目方案用 Okapi 标准 1.5/0.75，BEIR 官方 Lucene 基线用 Anserini 的 0.9/0.4；
    stem=True 加 Porter 词干化，进一步逼近官方分析链）"""
    n = len(doc_ids)
    postings = {}
    doc_len = []
    stemmer = PorterStemmer() if stem else None

    def tok(text):
        toks = tokenize(text)
        return [stemmer.stem(t) for t in toks] if stem else toks

    for d, text in enumerate(full_texts):
        tf = Counter(tok(text))
        doc_len.append(sum(tf.values()))
        for term, count in tf.items():
            postings.setdefault(term, []).append([d, count])
    avgdl = sum(doc_len) / max(n, 1)

    def search(qtext, k):
        cand = {}
        for t in tok(qtext):
            pl = postings.get(t)
            if not pl:
                continue
            idf = math.log(1 + (n - len(pl) + 0.5) / (len(pl) + 0.5))
            for d, f in pl:
                dl = doc_len[d]
                cand[d] = cand.get(d, 0.0) + idf * f * (k1 + 1) / \
                    (f + k1 * (1 - b + b * dl / avgdl))
        ranked = sorted(cand.items(), key=lambda kv: -kv[1])[:k]
        return [doc_ids[d] for d, _ in ranked]
    return search


def make_hybrid_searcher(pool_search, bm25_search, rrf_k=60):
    """RRF 融合两个文档排名列表（与 indexes/hybrid.py 的融合方式一致）。
    search(qtext, qv, k)：dense 路用查询向量，稀疏路用查询文本。"""
    def search(qtext, qv, k):
        dense_rank = pool_search(qv, RETRIEVAL_K)
        sparse_rank = bm25_search(qtext, RETRIEVAL_K)
        scores = {}
        for lst in (dense_rank, sparse_rank):
            for rank, did in enumerate(lst, start=1):
                scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [d for d, _ in ranked]
    return search


# ---------- 评测 ----------

def evaluate(searcher, queries, qrels, qid_list, mode="vec"):
    """逐题评测：NDCG@10 + Recall@5/10/20/50/100，返回逐题结果列表

    mode："text" 检索器只吃查询文本（bm25）；"vec" 只吃查询向量（向量方案）；
    "both" 两者都吃（hybrid）。
    """
    results = []
    for qid in qid_list:
        qtext = queries[qid]
        if mode == "text":
            ranked = searcher(qtext, RETRIEVAL_K)
        elif mode == "both":
            qv = np.array([OLLAMA_CLIENT.embeddings.create(
                model="bge-m3", input=qtext).data[0].embedding], dtype=np.float32)
            faiss.normalize_L2(qv)
            ranked = searcher(qtext, qv, RETRIEVAL_K)
        else:
            qv = np.array([OLLAMA_CLIENT.embeddings.create(
                model="bge-m3", input=qtext).data[0].embedding], dtype=np.float32)
            faiss.normalize_L2(qv)
            ranked = searcher(qv, RETRIEVAL_K)
        # 官方口径指标：逐题计算；NDCG 的理想折损取该题全部相关文档的 score
        # 降序（qrels 全集），而非检索结果内部（否则漏检越多 NDCG 越高，系统性偏差）
        rel_total = len(qrels[qid])
        ranked_scores = [qrels[qid].get(d, 0) for d in ranked]
        ndcg = ndcg_at_k(ranked_scores, NDCG_K, ideal_scores=list(qrels[qid].values()))
        recall = {}
        hits = [1 for s in ranked_scores if s > 0]
        for k in K_LIST:
            recall[k] = sum(1 for s in ranked_scores[:k] if s > 0) / rel_total if rel_total else 0.0
        results.append({"qid": qid, "ndcg": ndcg, "recall": recall,
                        "rel_total": rel_total, "hits": len(hits)})
    return results


def summarize(results, method, ds):
    n = len(results)
    avg_ndcg = sum(r["ndcg"] for r in results) / n
    out = [f"【{method} / {ds}】{n} 题（全量 qrels，官方口径）",
           f"  NDCG@10  = {avg_ndcg:.4f}"]
    for k in K_LIST:
        out.append(f"  Recall@{k:<3} = {sum(r['recall'][k] for r in results) / n:.4f}")
    return out, avg_ndcg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all")
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--score2-only", action="store_true",
                    help="仅把 score=2 视为相关（对标'官方只认强相关'的口径假设）")
    args = ap.parse_args()
    method, ds = args.method, args.dataset
    files = ds_files(ds)

    print(f"文档级官方口径评测：method={method} dataset={ds}"
          f"{'（仅 score=2 相关）' if args.score2_only else ''}")
    queries = load_queries(files["queries"])
    qrels = load_qrels(files["qrels"])
    if args.score2_only:
        qrels = {q: {d: s for d, s in v.items() if s >= 2}
                 for q, v in qrels.items()}
        qrels = {q: v for q, v in qrels.items() if v}
    qid_list = [q for q in queries if q in qrels]
    print(f"查询 {len(queries)} 条，有标注 {len(qid_list)} 条（全量 qrels 不截断）")

    chunk_texts, chunk_doc_ids = load_chunks(files["chunks"])
    print(f"块文件 {len(chunk_texts)} 块")

    corpus = load_corpus(files["corpus"])
    doc_ids_all = sorted(corpus.keys())
    full_texts = [corpus[d] for d in doc_ids_all]
    print(f"语料 {len(doc_ids_all)} 篇文档（标题+正文）")

    block_vecs = load_block_vectors(files["block_index"], chunk_texts, len(chunk_texts))

    # 构造各方法检索器（bm25 用查询文本，其余用查询向量）
    pool = make_pool_searcher(block_vecs, chunk_doc_ids)
    bm25 = make_bm25_doc_searcher(doc_ids_all, full_texts)
    block_index = faiss.IndexFlatIP(block_vecs.shape[1])
    block_index.add(np.ascontiguousarray(block_vecs))
    block2doc = make_block_to_doc_searcher(block_index, chunk_doc_ids)

    approx = {}
    for m, fname in (("ivf", "faiss_ivf"), ("hnsw", "faiss_hnsw"), ("pq", "faiss_pq")):
        path = os.path.join(DATA_ROOT, "vector_store", fname, "index.faiss")
        if not os.path.exists(path):
            continue
        idx = faiss.read_index(path)
        if idx.ntotal != len(chunk_texts):
            print(f"  [跳过 {m}] 索引块数 {idx.ntotal} 与块文件 {len(chunk_texts)} 不一致"
                  f"（当前 vector_store 不是 {ds} 数据）")
            continue
        if m == "hnsw":
            idx.hnsw.efSearch = EF_SEARCH        # HNSW 的搜索宽度在 hnsw 子结构上
        else:
            idx.nprobe = NPROBE                  # IVF / IVFPQ 的簇探测数
        approx[m] = make_block_to_doc_searcher(idx, chunk_doc_ids)

    bm25_anserini = make_bm25_doc_searcher(doc_ids_all, full_texts, k1=0.9, b=0.4)
    bm25_porter = make_bm25_doc_searcher(doc_ids_all, full_texts, k1=0.9, b=0.4, stem=True)
    # 无标题版语料：对标 BEIR 官方基线"语料是否含标题"的影响
    corpus_no_title = load_corpus(files["corpus"], include_title=False)
    doc_ids_nt = sorted(corpus_no_title.keys())
    bm25_nottitle = make_bm25_doc_searcher(doc_ids_nt, [corpus_no_title[d] for d in doc_ids_nt])

    methods = {
        "dense": ("vec", lambda qv, k: make_dense_searcher(doc_ids_all, full_texts, ds)(qv, k)),
        "doc_dense": ("vec", lambda qv, k: pool(qv, k)),
        "parent_doc": ("vec", lambda qv, k: block2doc(qv, k)),
        "bm25": ("text", lambda qtext, k: bm25(qtext, k)),
        "bm25_anserini": ("text", lambda qtext, k: bm25_anserini(qtext, k)),
        "bm25_porter": ("text", lambda qtext, k: bm25_porter(qtext, k)),
        "bm25_nottitle": ("text", lambda qtext, k: bm25_nottitle(qtext, k)),
        "hybrid": ("both", lambda qtext, qv, k: make_hybrid_searcher(pool, bm25)(qtext, qv, k)),
    }
    for m, s in approx.items():
        methods[m] = ("vec", (lambda qv, k, s=s: s(qv, k)))

    run_list = list(methods) if method == "all" else [method]
    # dense 的检索器惰性构建（避免每次都重新向量化；同一次运行内构建一次）
    dense_searcher = [None]
    csv_rows = []
    all_ndcg = {}
    for m in run_list:
        print(f"\n===== {m} =====")
        if m == "dense":
            if dense_searcher[0] is None:
                dense_searcher[0] = make_dense_searcher(doc_ids_all, full_texts, ds)
            mode, searcher = "vec", dense_searcher[0]
        else:
            mode, searcher = methods[m]
        results = evaluate(searcher, queries, qrels, qid_list, mode=mode)
        lines, avg_ndcg = summarize(results, m, ds)
        for line in lines:
            print(line)
        all_ndcg[m] = avg_ndcg
        for r in results:
            csv_rows.append([m, ds, r["qid"], r["ndcg"],
                             *[r["recall"][k] for k in K_LIST], r["rel_total"], r["hits"]])

    # csv 落盘
    csv_path = os.path.join(RESULTS_DIR, f"doc_level_{method}_{ds}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方法", "数据集", "题id", "NDCG@10",
                    *[f"Recall@{k}" for k in K_LIST], "相关文档数", "命中数"])
        w.writerows(csv_rows)
    print(f"\n逐题结果已保存 {csv_path}")

    # 与官方基线对照（仅 nfcorpus）
    if ds == "nfcorpus":
        print("\nBEIR 官方基线对照（nfcorpus NDCG@10）：")
        for name, v in BASELINES_NFCORPUS:
            print(f"  {name:<18} {v:.3f}")
        print("  本实验各方法：")
        for m, v in all_ndcg.items():
            print(f"    {m:<18} {v:.4f}")


if __name__ == "__main__":
    main()
