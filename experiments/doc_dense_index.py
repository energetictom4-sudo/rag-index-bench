# -*- coding: utf-8 -*-
"""
实验第 4 步的预备实现：文档级索引 + "先搜文档再取块"（暂放 experiments/，不动正式代码）

实验结论（详见 README.md / 实验总结报告.txt）：
  方案 B（块向量平均池化）与方案 A（整文档向量化）效果相当（NDCG@10 差距 <0.01），
  且零向量化成本 —— 推荐正式方案采用池化，故本文件默认 MODE="pool"。

接口与 indexes/dense.py 完全一致（build_index / search），确认方案后：
  1. 把本文件复制为 indexes/doc_dense.py（复制后文件名即方案名）；
  2. 主程序 rag_demo.py 配置区把 INDEX_METHOD 改为 "doc_dense"、N_RESULTS 改为 10；
  3. "先搜文档再取块"的取块逻辑已实现在 search() 中：
     文档级检索命中 Top-K 文档后，把该文档的全部块按原顺序拼成上下文返回，
     直接喂给 rag_demo.py 的 generate_answer（无需改主程序）。

MODE 可选：
  "pool"  默认：复用现有块向量按 doc_id 平均池化（零向量化成本，实验推荐）
  "full"  整文档重新向量化（复用 experiments/cache 的全文向量缓存）

本文件只读现有数据（块向量、块文件、实验缓存），构建的索引落盘在独立子目录
faiss_doc_pool/（或 faiss_doc_full/），与现有 faiss_dense/ 互不影响。
"""

import json
import os
import sys

import faiss
import numpy as np

from common import (BLOCK_INDEX_FILE, CACHE_DIR, CHUNKS_FILE, CORPUS_FILE,
                    cached_embed, load_corpus_fulltexts)

MODE = "pool"            # "pool"=块向量池化（推荐） / "full"=整文档向量化
STORE_DIR_NAME = "faiss_doc_pool" if MODE == "pool" else "faiss_doc_full"

# 文档→块映射与块文本：取块环节用（块文本来自 beir_chunks.json，顺序即块编号）
_doc_chunks_cache = None


def _load_doc_chunks():
    """加载"文档→块"映射：{doc_id: [块文本列表（按原顺序）]}，进程内只读一次"""
    global _doc_chunks_cache
    if _doc_chunks_cache is not None:
        return _doc_chunks_cache
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunk_items = json.load(f)
    mapping = {}
    for item in chunk_items:
        mapping.setdefault(item["doc_id"], []).append(item["text"])
    _doc_chunks_cache = mapping
    return mapping


def _store_dir(config):
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def build_index(chunks, config):
    """构建文档级索引：池化（或全文向量化）→ 归一化 → IndexFlatIP → 落盘

    chunks 参数为接口兼容保留（与 indexes/dense.py 一致），文档级建库不依赖它。
    """
    print(f"[doc_dense] 构建文档级索引（MODE={MODE}）...")

    if MODE == "pool":
        # 方案B：复用现有块向量按 doc_id 平均池化（零向量化成本）
        block_index = faiss.read_index(BLOCK_INDEX_FILE)
        block_vecs = block_index.reconstruct_n(0, block_index.ntotal)
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            chunk_doc_ids = [item["doc_id"] for item in json.load(f)]
        if len(chunk_doc_ids) != block_index.ntotal:
            raise RuntimeError("块文件与块向量数量不一致，请重新生成（见 doc_retrieval_pool.py）")
        doc_ids = sorted(set(chunk_doc_ids))
        matrix = np.zeros((len(doc_ids), block_vecs.shape[1]), dtype=np.float32)
        for i, did in enumerate(doc_ids):
            mask = [d == did for d in chunk_doc_ids]
            matrix[i] = block_vecs[mask].mean(axis=0)
    else:
        # 方案A：整文档向量化（优先复用实验缓存，缺失部分自动补算）
        doc_ids, full_texts = load_corpus_fulltexts()
        matrix = cached_embed(full_texts, doc_ids, "full_doc", "全文")

    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(store_dir, "index.faiss"))
    with open(os.path.join(store_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print(f"[doc_dense] 已存入 {len(doc_ids)} 篇文档到 {store_dir}")


def _load_store(config):
    """加载文档级索引与文档 id 列表"""
    store_dir = _store_dir(config)
    index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
    with open(os.path.join(store_dir, "doc_ids.json"), encoding="utf-8") as f:
        doc_ids = json.load(f)
    return index, doc_ids


def search(question, config):
    """检索：问题向量化 → Top-K 文档 → 取块拼上下文（"先搜文档再取块"）

    返回每篇命中文档的完整上下文（该文档全部块按原顺序拼接），
    列表长度 = config["n_results"]，直接供 rag_demo.py 的 generate_answer 使用。
    """
    # 1. 问题向量化 + 归一化
    response = config["ollama_client"].embeddings.create(
        model=config["embedding_model"], input=question)
    query = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # 2. 文档级检索 Top-K 文档
    index, doc_ids = _load_store(config)
    k = min(config["n_results"], len(doc_ids))
    _, ids = index.search(query, k)

    # 3. 取块：命中文档的全部块按原顺序拼成上下文
    doc_chunks = _load_doc_chunks()
    contexts = []
    for i in ids[0]:
        did = doc_ids[i]
        blocks = doc_chunks.get(did, [])
        contexts.append("\n".join(blocks) if blocks else f"[文档 {did} 无块文本]")
    return contexts


# --- 自检：直接运行本文件，验证建库与"先搜文档再取块"链路可用 ---
if __name__ == "__main__":
    import rag_demo
    config = dict(rag_demo.INDEX_CONFIG)
    config["n_results"] = 3
    # 自检时改用绝对路径落盘（主程序传的相对路径依赖运行目录，自检不应产生漂移文件）
    config["storage_path"] = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "vector_store")

    if "--skip-build" in sys.argv:
        print("已指定 --skip-build，跳过建库，直接检索现有索引")
    else:
        build_index(None, config)

    demo_q = "Do Cholesterol Statin Drugs Cause Breast Cancer?"
    results = search(demo_q, config)
    print(f"\n===== 检索结果（问题：{demo_q}）=====")
    for i, ctx in enumerate(results, start=1):
        print(f"\n--- 文档{i}（前150字符）---\n{ctx[:150]}")
