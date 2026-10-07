"""
索引方案：IVF + PQ 乘积量化压缩索引（pq）
阶段三存储层优化方案三：在 IVF 聚类倒排的基础上，用乘积量化（Product Quantization）
压缩向量存储，把"浮点存储"的代价打下去（长期定位：面向存储效率的索引层优化）。

与 ivf（IndexIVFFlat）的唯一差异：ivf 倒排表里存的是完整 1024 维 float32 向量
（每块 4096 字节），pq 把每块向量切成 M 个子段、每个子段用量化码本里最近的码字编号
表示（M=32、8 位码时每块仅 32 字节，向量部分压缩 128 倍）。检索时用查表法
（ADC 非对称距离计算）算近似距离，召回略有损失，换来存储与距离计算的双重压缩。

实现说明：
  向量零成本复用：从 faiss_dense/index.faiss（dense 方案的块向量索引）reconstruct 提取
  全部块向量，不再重新向量化；本方案只落盘 faiss_pq/ 子目录（IVFPQ 索引 + 文本）。
  度量与 dense/ivf/hnsw 一致：向量 L2 归一化后内积即余弦相似度（METRIC_INNER_PRODUCT）。
  压缩比-召回-延迟的完整权衡扫描见 experiments/pq_curve.py。
"""

import json
import os
import sys

import faiss
import numpy as np

# 本方案在存储根目录下的子目录名，与 dense/ivf/hnsw 的子目录互不影响
STORE_DIR_NAME = "faiss_pq"

# 块向量来源：dense 方案的索引（复用其向量，零向量化成本）
DENSE_STORE_DIR_NAME = "faiss_dense"

# ================ 参数配置区 ================

NLIST = 256     # 聚类簇数（与 ivf 一致，保证对比时只有 PQ 一个变量）
M = 32          # PQ 子段数：1024 维切成 M 段，每段独立量化（越小压缩越狠、量化越粗）
N_BITS = 8      # 每个子段码字位数：8 位 = 每子段 256 个码字（标准配置）
NPROBE = 32     # 查询时搜索的最近簇数（与 ivf 甜蜜点一致）

# 已加载的索引缓存：避免每次检索都从磁盘重新读取
_cache = {}


def _store_dir(config: dict) -> str:
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def _dense_index_file(config: dict) -> str:
    """dense 方案的块向量索引文件路径（只读复用，不修改）"""
    return os.path.join(config["storage_path"], DENSE_STORE_DIR_NAME, "index.faiss")


def build_index(chunks: list, config: dict) -> None:
    """构建索引：从 dense 索引提取全部块向量 → 训练 IVF 聚类 + PQ 码本 → 写入本方案目录"""
    texts = [chunk.page_content for chunk in chunks]
    print(f"[pq] 从 dense 索引提取块向量并构建 IVFPQ 索引（nlist={NLIST}，M={M}，{N_BITS} 位）...")

    # 1. 提取 dense 的块向量（reconstruct 顺序与 dense 建库时 add 顺序一致，对应 texts 下标）
    dense_index = faiss.read_index(_dense_index_file(config))
    matrix = np.array(dense_index.reconstruct_n(0, dense_index.ntotal), dtype=np.float32)
    if len(matrix) != len(texts):
        raise RuntimeError(
            f"[pq] 向量数({len(matrix)})与块数({len(texts)})不一致："
            f"faiss_dense 索引可能过期，请先重建 dense 方案再构建 PQ"
        )

    # 2. 向量 L2 归一化（与 dense 一致：内积即余弦相似度；dense 已归一化，此处幂等保险）
    faiss.normalize_L2(matrix)

    # 3. 构建 IVFPQ 索引：train 同时训练聚类（粗量化）与 PQ 码本（细量化）
    d = matrix.shape[1]
    quantizer = faiss.IndexFlatIP(d)   # 聚类用粗量化器，度量与 ivf 一致
    index = faiss.IndexIVFPQ(quantizer, d, NLIST, M, N_BITS, faiss.METRIC_INNER_PRODUCT)
    print(f"[pq] 正在训练聚类 + PQ 码本（{NLIST} 簇，{M} 子段，每段 {2 ** N_BITS} 码字）...")
    index.train(matrix)
    index.add(matrix)

    # 4. 落盘：索引文件 + 文本文件
    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(store_dir, "index.faiss"))
    with open(os.path.join(store_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    _cache.clear()

    # 5. 报告压缩情况（向量部分：float32 全量 vs PQ 码，目录整体含 texts.json）
    raw_bytes = matrix.nbytes
    pq_bytes = matrix.shape[0] * (M * N_BITS // 8)
    index_file = os.path.join(store_dir, "index.faiss")
    print(f"[pq] 已存入 {len(texts)} 个片段到 {store_dir}"
          f"（nlist={NLIST}，M={M}，评测 nprobe={NPROBE}）")
    print(f"[pq] 向量存储：{raw_bytes / 1024 / 1024:.1f} MB → {pq_bytes / 1024 / 1024:.2f} MB"
          f"（压缩 {raw_bytes / pq_bytes:.0f} 倍）；索引文件 {os.path.getsize(index_file) / 1024 / 1024:.2f} MB")


def _load_store(config: dict) -> tuple:
    """加载 IVFPQ 索引与文本（带缓存，避免每次检索都读盘）"""
    store_dir = _store_dir(config)
    if store_dir not in _cache:
        index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        with open(os.path.join(store_dir, "texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        _cache[store_dir] = (index, texts)
    return _cache[store_dir]


def search(question: str, config: dict) -> list[str]:
    """检索：问题向量化 → 在 IVFPQ 索引中搜索最近的 nprobe 个簇（查表近似距离）"""
    # 1. 问题向量化 + 归一化
    response = config["ollama_client"].embeddings.create(
        model=config["embedding_model"],
        input=question
    )
    query = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # 2. 检索：IVFPQ 同样先设置 nprobe（搜索的簇数），再执行搜索
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
