# -*- coding: utf-8 -*-
"""
索引方案：父文档索引（parent_doc）—— 小块检索、大块返回

思路：解决"小块精确召回 vs 大块上下文完整"的矛盾（LangChain ParentDocumentRetriever
的标准做法，面试常考）：
  - 子块（小块，约500字符）：向量化的最小粒度，语义聚焦、匹配精确
  - 父文档（大块，整篇文档）：命中子块后，返回其所属文档的全部块按原顺序拼接
检索时先搜子块，再把命中的子块映射回所属文档，返回文档级完整上下文。

与 doc_dense 的对照实验（本方案的研究价值）：
  - doc_dense：文档级向量（块向量平均池化）→ 搜文档 → 返回文档
  - parent_doc：块级向量（原始块向量）→ 搜块 → 映射文档 → 返回文档
  两者返回的都是文档级上下文，唯一区别是检索粒度：池化平均会稀释语义
  （一篇文档的多个主题被平均成一个向量），块级向量则保留每个主题的原始语义。
  实验假设：块级检索的召回率应 ≥ 文档级检索，评测数据见项目 README 对比表。

实现说明：
  - 子块向量直接复用 dense 方案的索引（vector_store/faiss_dense/），零向量化成本；
    "块→文档"映射来自 beir_chunks.json（块编号即文件下标，与块向量顺序一致）
  - 多块命中同一文档时保序去重（最高分的块先触发该文档），不足 n_results 个文档
    则继续向下取块补足（与 doc_dense 的返回数量口径一致，保证精确率分母公平）
  - 本方案落盘 meta.json（文档id列表与元信息），子块索引仍在 faiss_dense/ 子目录
"""

import json
import os
import sys

# experiments 目录加入搜索路径：本方案复用 experiments/common.py 的路径常量
# （CHUNKS_FILE、BLOCK_INDEX_FILE），与 doc_dense.py 的导入方式一致
_EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from common import BLOCK_INDEX_FILE, CHUNKS_FILE   # noqa: E402

import faiss   # noqa: E402
import numpy as np   # noqa: E402

# ================ 本方案专属参数 ================
SUB_TOP = 200    # 子块检索候选数：去重后要凑够 n_results 个文档，
                 # 每题相关文档平均约38篇，Top-200 块通常覆盖 50+ 个文档

STORE_DIR_NAME = "faiss_parent_doc"

# 块向量索引与"块→文档"映射：进程内只读一次
_block_index_cache = None
_doc_chunks_cache = None


def _load_block_index():
    """加载子块向量索引（faiss_dense 的块向量，只读复用，进程内缓存）"""
    global _block_index_cache
    if _block_index_cache is None:
        _block_index_cache = faiss.read_index(BLOCK_INDEX_FILE)
    return _block_index_cache


def _load_doc_chunks():
    """加载"文档→块"映射与"块编号→文档id"列表（来自 beir_chunks.json，只读一次）

    返回 (doc_chunks, chunk_doc_ids)：
      doc_chunks      {doc_id: [块文本列表（按原顺序）]}
      chunk_doc_ids   下标即块编号，值为来源文档id（块编号与块向量行号一致）
    """
    global _doc_chunks_cache
    if _doc_chunks_cache is not None:
        return _doc_chunks_cache
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunk_items = json.load(f)
    doc_chunks = {}
    chunk_doc_ids = []
    for item in chunk_items:
        doc_chunks.setdefault(item["doc_id"], []).append(item["text"])
        chunk_doc_ids.append(item["doc_id"])
    _doc_chunks_cache = (doc_chunks, chunk_doc_ids)
    return _doc_chunks_cache


def _store_dir(config: dict) -> str:
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def build_index(chunks: list, config: dict) -> None:
    """构建父文档索引：校验子块向量索引可用 → 落盘文档id列表与元信息

    chunks 参数用于校验块数一致性（子块向量必须与当前块数据同步），
    实际向量复用 dense 方案的索引，本方案零向量化成本。
    """
    print("[parent_doc] 构建父文档索引（子块向量复用 faiss_dense）...")

    # 1. 校验子块向量索引与当前块数据一致（块数对不上说明 beir_chunks.json
    #    被重新生成过，需要先重建 dense 索引，避免"张冠李戴"的静默错误）
    _, chunk_doc_ids = _load_doc_chunks()
    block_index = _load_block_index()
    if block_index.ntotal != len(chunk_doc_ids):
        raise RuntimeError(
            f"块向量索引（{block_index.ntotal} 行）与块文件（{len(chunk_doc_ids)} 块）"
            f"数量不一致：请先把 INDEX_METHOD 切回 dense 重跑建库，再回来建 parent_doc"
        )
    if len(chunk_doc_ids) != len(chunks):
        raise RuntimeError(
            f"主程序传入 {len(chunks)} 个块，块文件含 {len(chunk_doc_ids)} 块，"
            f"数量不一致：请检查 beir_chunks.json 是否与评测脚本加载的数据同源"
        )

    # 2. 落盘元信息：有序文档id列表（sorted 保证确定性，与 doc_dense 一致）
    doc_ids = sorted(set(chunk_doc_ids))
    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    with open(os.path.join(store_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_blocks": len(chunk_doc_ids),
            "n_docs": len(doc_ids),
            "doc_ids": doc_ids,
        }, f, ensure_ascii=False)
    print(f"[parent_doc] 已就绪：{len(chunk_doc_ids)} 个子块 → {len(doc_ids)} 篇父文档，"
          f"元信息落盘 {store_dir}")


def search(question: str, config: dict) -> list[str]:
    """检索：问题向量化 → 块级 Top-SUB_TOP → 映射文档去重 → 返回文档完整上下文

    返回列表长度 ≤ config["n_results"]（极端情况下 Top-SUB_TOP 块覆盖的
    文档数不足时宁缺勿凑，此时召回率口径不受影响、精确率略吃亏）。
    """
    # 1. 问题向量化 + 归一化（与 dense 方案相同的向量化流程）
    response = config["ollama_client"].embeddings.create(
        model=config["embedding_model"], input=question)
    query = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # 2. 块级检索：取 Top-SUB_TOP 个最相似子块
    block_index = _load_block_index()
    doc_chunks, chunk_doc_ids = _load_doc_chunks()
    k = min(SUB_TOP, block_index.ntotal)
    _, ids = block_index.search(query, k)

    # 3. 块编号 → 文档id：保序去重（最高分块先触发其文档，文档顺序按最佳匹配块排序）
    hit_docs = []
    for block_id in ids[0]:
        did = chunk_doc_ids[block_id]
        if did not in hit_docs:
            hit_docs.append(did)
        if len(hit_docs) >= config["n_results"]:
            break

    # 4. 返回每篇命中文档的完整上下文（全部块按原顺序拼接）
    contexts = []
    for did in hit_docs:
        blocks = doc_chunks.get(did, [])
        contexts.append("\n".join(blocks) if blocks else f"[文档 {did} 无块文本]")
    return contexts


# --- 自检：直接运行本文件，验证"小块检索→大块返回"链路可用 ---
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
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            items = json.load(f)
        chunks = [Document(page_content=item["text"], metadata={"chunk": item["id"]})
                  for item in items]
        build_index(chunks, config)

    demo_q = "Do Cholesterol Statin Drugs Cause Breast Cancer?"
    results = search(demo_q, config)
    print(f"\n===== 检索结果（方案：parent_doc，问题：{demo_q}）=====")
    for i, ctx in enumerate(results, start=1):
        print(f"\n--- 父文档{i}（前150字符）---\n{ctx[:150]}")
