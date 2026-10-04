"""
索引方案：HNSW 分层可导航小世界图索引（hnsw）
阶段三存储层优化方案二：把 dense 的 IndexFlatIP 暴力检索换成 HNSW 图索引近似检索（ANN）。

与 IVF（聚类倒排）的对比意义：
  IVF 是"空间划分"路线（k-means 分簇，只搜附近簇）；HNSW 是"图导航"路线
  （建多层近邻图，从高层粗跳、低层精找）。两种主流 ANN 路线的召回率-延迟
  曲线合并对比见 experiments/ann_curve.py。

实现说明：
  向量零成本复用：从 faiss_dense/index.faiss（dense 方案的块向量索引）reconstruct 提取
  全部块向量，不再重新向量化；本方案只落盘 faiss_hnsw/ 子目录（图索引 + 文本）。
  度量与 dense/ivf 一致：向量 L2 归一化后内积即余弦相似度（METRIC_INNER_PRODUCT）。
  ef_search 对召回-延迟的完整权衡曲线见 experiments/ann_curve.py 微基准。

本机环境坑记录（2026-10-04，已修复）：
  曾发现本机任何 C++ HNSW 实现（faiss 的 IndexHNSWFlat、chroma-hnswlib）建图必然段错误，
  根因是 anaconda 根目录的 2020 年旧版 VC 运行时 dll 劫持了全进程（详见工作交接文档
  第六节踩坑记录）。已将旧 dll 改名备份为 .bak2020，进程自动改用系统新版运行时。
"""

import json
import os
import sys

import faiss
import numpy as np

# 本方案在存储根目录下的子目录名，与 dense 的 faiss_dense/、ivf 的 faiss_ivf/ 互不影响
STORE_DIR_NAME = "faiss_hnsw"

# 块向量来源：dense 方案的索引（复用其向量，零向量化成本）
DENSE_STORE_DIR_NAME = "faiss_dense"

# ================ 参数配置区 ================

M = 32                # 图中每个节点连接数上限（越大图越密、召回越高、建图越慢、索引越大）
EF_CONSTRUCTION = 64  # 建图时的搜索宽度（越大图质量越高，只影响建图耗时不影响检索）
EF_SEARCH = 64        # 检索时的搜索宽度（1~数百；越大越接近暴力检索的召回，也越慢）

# 已加载的索引缓存：避免每次检索都从磁盘重新读取
_cache = {}


def _store_dir(config: dict) -> str:
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def _dense_index_file(config: dict) -> str:
    """dense 方案的块向量索引文件路径（只读复用，不修改）"""
    return os.path.join(config["storage_path"], DENSE_STORE_DIR_NAME, "index.faiss")


def build_index(chunks: list, config: dict) -> None:
    """构建索引：从 dense 索引提取全部块向量 → 逐点插入构建 HNSW 多层图 → 写入本方案目录"""
    texts = [chunk.page_content for chunk in chunks]
    print(f"[hnsw] 从 dense 索引提取块向量并构建 HNSW 图索引（M={M}）...")

    # 1. 提取 dense 的块向量（reconstruct 顺序与 dense 建库时 add 顺序一致，对应 texts 下标）
    dense_index = faiss.read_index(_dense_index_file(config))
    matrix = np.array(dense_index.reconstruct_n(0, dense_index.ntotal), dtype=np.float32)
    if len(matrix) != len(texts):
        raise RuntimeError(
            f"[hnsw] 向量数({len(matrix)})与块数({len(texts)})不一致："
            f"faiss_dense 索引可能过期，请先重建 dense 方案再构建 HNSW"
        )

    # 2. 向量 L2 归一化（与 dense 一致：内积即余弦相似度；dense 已归一化，此处幂等保险）
    faiss.normalize_L2(matrix)

    # 3. 构建 HNSW 图索引：逐点插入，按内积（余弦）度量连边分层
    d = matrix.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    print(f"[hnsw] 正在插入 {len(matrix)} 个向量建图（efConstruction={EF_CONSTRUCTION}）...")
    index.add(matrix)

    # 4. 落盘：索引文件 + 文本文件
    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(store_dir, "index.faiss"))
    with open(os.path.join(store_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    _cache.clear()
    print(f"[hnsw] 已存入 {len(texts)} 个片段到 {store_dir}（M={M}，评测 ef_search={EF_SEARCH}）")


def _load_store(config: dict) -> tuple:
    """加载 HNSW 索引与文本（带缓存，避免每次检索都读盘）"""
    store_dir = _store_dir(config)
    if store_dir not in _cache:
        index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        with open(os.path.join(store_dir, "texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        _cache[store_dir] = (index, texts)
    return _cache[store_dir]


def search(question: str, config: dict) -> list[str]:
    """检索：问题向量化 → 在 HNSW 图上按 ef_search 宽度贪心导航取相似片段"""
    # 1. 问题向量化 + 归一化
    response = config["ollama_client"].embeddings.create(
        model=config["embedding_model"],
        input=question
    )
    query = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # 2. 检索：ef_search 是运行时参数（不落盘），每次检索前设置
    index, texts = _load_store(config)
    index.hnsw.efSearch = EF_SEARCH
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
