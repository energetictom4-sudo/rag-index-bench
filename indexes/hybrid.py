# -*- coding: utf-8 -*-
"""
索引方案：混合检索（hybrid）—— BM25 + 稠密向量，RRF 分数融合

思路：BM25 擅长"词面关键词"匹配（精确、可解释），dense 擅长"语义"匹配
（能抓住换了说法的同义表达），两者互补。本方案同时检索两种索引，
把两个排序结果用 RRF（倒数排名融合）合并为一个最终排序——
这是工业界 RAG 的标准做法（如 Weaviate、Elasticsearch 的混合检索）。

RRF 原理（面试常考）：
  score(d) = Σ 1 / (k + rank_i(d))
    rank_i(d) = 文档 d 在第 i 个排序结果中的名次（从 1 开始）
    k = 60（经验常数，降低"第一名碾压一切"的权重，让各来源的靠前名次都能发声）
  特点：
    - 只依赖名次、不依赖原始分数——BM25 与向量相似度两个量纲不同的分数
      无法直接相加，RRF 通过"排名"把两者统一到同一尺度
    - 文档在多个来源都靠前 → 名次小 → 得分高；只在一个来源靠前 → 得分打折
    - 无需求解任何权重参数，业界公认稳

实现方式：组合器模式——本方案不重复造轮子，直接调用 indexes/bm25.py 与
indexes/dense.py 的 build_index/search，两个子方案的索引数据各自落盘在自己的
子目录（bm25/、faiss_dense/），本方案只做融合。
"""

import os
import sys

# 项目根目录加入搜索路径：直接以文件路径运行本脚本自检时，indexes 包不在
# sys.path 中（与 doc_dense.py 插 experiments 路径同理）
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from indexes import bm25 as _bm25   # noqa: E402  稀疏检索子方案
from indexes import dense as _dense  # noqa: E402  稠密向量子方案

# ================ 本方案专属参数 ================
RRF_K = 60        # RRF 常数：1/(k+名次) 中的 k，60 为业界常用值
RRF_TOP = 60      # 每个子方案取前多少名参与融合（候选池大小，越大召回越全、噪声越多）

# dense 子方案的存储目录（复用 dense.py 的配置，建库前检查是否已有可用索引）
_DENSE_STORE_DIR = "faiss_dense"


def _dense_store_ready(chunks: list, config: dict) -> bool:
    """检查 dense 子方案索引是否已存在且与当前块数据一致（避免重复向量化）

    dense 建库要对全部块做向量化（本地 CPU 约 220 秒），若 9 月评测时已建好、
    块数据未变，直接复用即可——检查 texts.json 的块数与当前 chunks 是否一致。
    """
    store_dir = os.path.join(config["storage_path"], _DENSE_STORE_DIR)
    index_file = os.path.join(store_dir, "index.faiss")
    texts_file = os.path.join(store_dir, "texts.json")
    if not (os.path.exists(index_file) and os.path.exists(texts_file)):
        return False
    import json
    with open(texts_file, encoding="utf-8") as f:
        texts = json.load(f)
    return len(texts) == len(chunks)


def build_index(chunks: list, config: dict) -> None:
    """构建混合索引：两个子方案各自建库（数据落盘在各自的子目录）

    - bm25 建库约 3 秒（纯倒排构建），直接重建
    - dense 建库约 220 秒（全量向量化），已有索引且块数一致时跳过
    """
    print("[hybrid] 构建混合索引（BM25 + dense，RRF 融合）...")

    _bm25.build_index(chunks, config)

    if _dense_store_ready(chunks, config):
        print("[hybrid] dense 子方案索引已存在且块数一致，跳过重建（复用 9 月评测时的向量）")
    else:
        _dense.build_index(chunks, config)

    print("[hybrid] 混合索引就绪（bm25/ 与 faiss_dense/ 两套子索引并存）")


def search(question: str, config: dict) -> list[str]:
    """检索：两个子方案各取 Top-RRF_TOP → RRF 融合 → 返回前 n_results 个片段"""
    # 1. 两个子方案各自检索（bm25 毫秒级；dense 含问题向量化，约 240 毫秒）
    bm25_results = _bm25.search(question, config)
    dense_results = _dense.search(question, config)

    # 2. RRF 融合：按名次计分合并两个排序（块文本做 key——两份子索引的文本
    #    都来自同一份 beir_chunks.json，顺序与内容一致，同一文本即同一块）
    fused = _rrf_fuse(
        [bm25_results[:RRF_TOP], dense_results[:RRF_TOP]], k=RRF_K
    )

    # 3. 取融合后前 n_results 个
    return [text for text, _ in fused[:config["n_results"]]]


def _rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """RRF 融合：score(d) = Σ 1/(k + 名次)，按得分降序返回 (文本, 得分) 列表"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, text in enumerate(ranked, start=1):   # 名次从 1 开始
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# --- 自检：直接运行本文件，验证融合检索链路可用 ---
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # 解决 Windows 控制台中文乱码
    import rag_demo   # 仅自检复用主程序配置（模块级不导入，保持方案脚本零依赖）
    from langchain_core.documents import Document

    config = dict(rag_demo.INDEX_CONFIG)
    config["n_results"] = 3
    # 自检时改用绝对路径落盘（主程序传的相对路径依赖运行目录，自检不应产生漂移文件）
    config["storage_path"] = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "vector_store")

    if "--skip-build" in sys.argv:
        print("已指定 --skip-build，跳过建库，直接检索现有索引")
    else:
        # 加载 BEIR 块文件（与主程序一致）
        chunks_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "beir_chunks.json")
        import json
        with open(chunks_file, encoding="utf-8") as f:
            items = json.load(f)
        chunks = [Document(page_content=item["text"], metadata={"chunk": item["id"]})
                  for item in items]
        build_index(chunks, config)

    demo_q = "Do Cholesterol Statin Drugs Cause Breast Cancer?"
    results = search(demo_q, config)
    print(f"\n===== 检索结果（方案：hybrid，问题：{demo_q}）=====")
    for i, text in enumerate(results, start=1):
        print(f"\n--- 片段{i}（前150字符）---\n{text[:150]}")
