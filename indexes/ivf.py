"""
索引方案：IVF 倒排文件索引（ivf）
阶段三存储层优化方案一：把 dense 的 IndexFlatIP 暴力检索换成 IVF 近似最近邻（ANN）检索。

实现说明：
  向量零成本复用：从 faiss_dense/index.faiss（dense 方案的块向量索引）reconstruct 提取
  全部块向量，不再重新向量化；本方案只落盘 faiss_ivf/ 子目录（IVF 索引 + 文本）。
  检索原理：k-means 把全部块向量聚成 nlist 个簇，查询时只搜索最近的 nprobe 个簇
  （nprobe 越小越省时、但召回损失越大；nprobe=nlist 时近似等于扫全部簇）。
  度量与 dense 一致：向量 L2 归一化后内积即余弦相似度，quantizer 也用 IndexFlatIP。
  nprobe 对召回-延迟的完整权衡曲线见 experiments/bench_retrieval.py 微基准。
"""

import json
import os
import sys

import faiss
import numpy as np

# 本方案在存储根目录下的子目录名，与 dense 的 faiss_dense/ 互不影响
STORE_DIR_NAME = "faiss_ivf"

# 块向量来源：dense 方案的索引（复用其向量，零向量化成本）
DENSE_STORE_DIR_NAME = "faiss_dense"

# ================ 参数配置区 ================

NLIST = 256      # 聚类簇数：k-means 把全部块向量分成多少个簇（影响建库与检索粒度）
NPROBE = 32      # 查询时搜索的最近簇数（1~NLIST；越大越接近暴力检索的召回，也越慢）

# 已加载的索引缓存：避免每次检索都从磁盘重新读取
_cache = {}


def _store_dir(config: dict) -> str:
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def _dense_index_file(config: dict) -> str:
    """dense 方案的块向量索引文件路径（只读复用，不修改）"""
    return os.path.join(config["storage_path"], DENSE_STORE_DIR_NAME, "index.faiss")


def build_index(chunks: list, config: dict) -> None:
    """构建索引：从 dense 索引提取全部块向量 → 训练 IVF 聚类 → 写入本方案目录"""
    texts = [chunk.page_content for chunk in chunks]
    print(f"[ivf] 从 dense 索引提取块向量并构建 IVF 索引（nlist={NLIST}）...")

    # 1. 提取 dense 的块向量（reconstruct 顺序与 dense 建库时 add 顺序一致，对应 texts 下标）
    dense_index = faiss.read_index(_dense_index_file(config))
    matrix = np.array(dense_index.reconstruct_n(0, dense_index.ntotal), dtype=np.float32)
    if len(matrix) != len(texts):
        raise RuntimeError(
            f"[ivf] 向量数({len(matrix)})与块数({len(texts)})不一致："
            f"faiss_dense 索引可能过期，请先重建 dense 方案再构建 IVF"
        )

    # 2. 向量 L2 归一化（与 dense 一致：内积即余弦相似度；dense 已归一化，此处幂等保险）
    faiss.normalize_L2(matrix)

    # 3. 构建 IVF 索引：quantizer 负责"查询向量找最近簇"，度量与主索引一致用内积
    d = matrix.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)
    print(f"[ivf] 正在训练 k-means 聚类（{NLIST} 个簇，{len(matrix)} 个向量）...")
    index.train(matrix)      # 聚类训练只用到向量本身
    index.add(matrix)        # 全部向量按簇分配存入倒排表

    # 4. 落盘：索引文件 + 文本文件
    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(store_dir, "index.faiss"))
    with open(os.path.join(store_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    _cache.clear()
    print(f"[ivf] 已存入 {len(texts)} 个片段到 {store_dir}（nlist={NLIST}，评测 nprobe={NPROBE}）")


def _load_store(config: dict) -> tuple:
    """加载 IVF 索引与文本（带缓存，避免每次检索都读盘）"""
    store_dir = _store_dir(config)
    if store_dir not in _cache:
        index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        with open(os.path.join(store_dir, "texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        _cache[store_dir] = (index, texts)
    return _cache[store_dir]


def search(question: str, config: dict) -> list[str]:
    """检索：问题向量化 → 在 IVF 索引中搜索最近的 nprobe 个簇取相似片段"""
    # 1. 问题向量化 + 归一化
    response = config["ollama_client"].embeddings.create(
        model=config["embedding_model"],
        input=question
    )
    query = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # 2. 检索：IVF 索引需先设置 nprobe（搜索的簇数），再执行搜索
    index, texts = _load_store(config)
    index.nprobe = NPROBE
    k = min(config["n_results"], len(texts))
    _, ids = index.search(query, k)
    return [texts[i] for i in ids[0]]


if __name__ == "__main__":
    """自检：验证本方案索引能正常检索（需先运行 python eval_retrieval.py 建库）"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import rag_demo

    config = dict(rag_demo.INDEX_CONFIG)
    config["n_results"] = 3
    try:
        results = search("federal blood donation regulations", config)
    except FileNotFoundError as e:
        print(f"[自检] 未找到索引文件（{e}），请先运行 python eval_retrieval.py 建库")
        sys.exit(1)
    print(f"[自检] 检索到 {len(results)} 个片段：")
    for t in results:
        print(f"  - {t[:60].replace(chr(10), ' ')}...")
