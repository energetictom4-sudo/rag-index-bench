"""
索引方案：稠密向量索引（dense）
这是最基础、最常用的索引方案：把每个切分片段向量化后存入FAISS向量索引，
检索时把问题向量化，在索引中找最相似的片段。

实现说明：
  向量库使用 FAISS（IndexFlatIP 内积索引，向量先做L2归一化后内积即余弦相似度）。
  原计划使用Chroma存储，但本机chromadb存在段错误问题，改用语义完全相同的FAISS实现。
  数据存储：config["storage_path"] 目录下的 faiss_dense/ 子目录（index.faiss + texts.json）

【写新索引方案的方法】
1. 复制本文件，改名为你的方案名（如 parent_doc.py），文件名即方案名
2. 按下面两个函数的接口约定改写内部逻辑
3. 在主程序 rag_demo.py 配置区把 INDEX_METHOD 改成你的文件名即可，主程序不用动
接口约定详见 indexes/README.md
"""

import json
import os

import faiss
import numpy as np

# 本方案在存储根目录下的子目录名，不同方案用不同子目录避免数据互相覆盖
STORE_DIR_NAME = "faiss_dense"

# 已加载的索引缓存：避免每次检索都从磁盘重新读取
_cache = {}


def _store_dir(config):
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def build_index(chunks, config):
    """构建索引：分批向量化所有片段，存入FAISS索引并落盘"""
    texts = [chunk.page_content for chunk in chunks]
    print(f"[dense] 正在生成向量并存入FAISS索引...")

    # 1. 分批向量化（避免一次请求过大），收集全部向量
    all_embeddings = []
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        batch = texts[i:end]
        response = config["ollama_client"].embeddings.create(
            model=config["embedding_model"],
            input=batch
        )
        all_embeddings.extend(item.embedding for item in response.data)
        print(f"[dense] 向量化进度：{end}/{len(texts)}")

    # 2. 构建FAISS索引：向量先L2归一化，内积检索等价于余弦相似度
    matrix = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    # 3. 落盘：索引文件 + 文本文件
    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(store_dir, "index.faiss"))
    with open(os.path.join(store_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    _cache.clear()
    print(f"[dense] 已存入 {len(texts)} 个片段到 {store_dir}")


def _load_store(config):
    """加载FAISS索引与文本（带缓存，避免每次检索都读盘）"""
    store_dir = _store_dir(config)
    if store_dir not in _cache:
        index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        with open(os.path.join(store_dir, "texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        _cache[store_dir] = (index, texts)
    return _cache[store_dir]


def search(question, config):
    """检索：把问题向量化后，在FAISS索引中找最相似的片段"""
    # 1. 问题向量化 + 归一化
    response = config["ollama_client"].embeddings.create(
        model=config["embedding_model"],
        input=question
    )
    query = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    # 2. 检索最相关的n个片段
    index, texts = _load_store(config)
    k = min(config["n_results"], len(texts))
    _, ids = index.search(query, k)
    return [texts[i] for i in ids[0]]
