# -*- coding: utf-8 -*-
"""
索引方案：BM25 稀疏检索（关键词倒排索引）

与 dense 方案（语义向量检索）形成经典对比：BM25 按"词面关键词"匹配，
不需要向量化、不依赖嵌入模型，速度快、可解释性强（能精确说出是哪些词
命中的），但抓不住"换了说法"的语义匹配（例如检索"高血压"不会命中只写
"血压升高"的片段）。

BM25 原理（面试常考，详见各函数注释）：
  1. 倒排索引：预处理时把每个词记录到"哪些文档出现过、出现几次"，
     检索时只算包含问题词的文档，避免全库扫描
  2. Okapi BM25 打分公式（k1=1.5, b=0.75 为标准参数）：
       IDF(t) = ln( 1 + (N - df + 0.5) / (df + 0.5) )
         N=文档总数，df=包含词t的文档数 → 词越稀有权重越高
       score(d,q) = Σ IDF(t) × tf×(k1+1) / (tf + k1×(1-b+b×dl/avgdl))
         tf=词在文档中的次数，dl=文档长度，avgdl=平均文档长度
         → 出现次数多加分，但边际递减；长文档被归一化惩罚

分词说明：
  英文按 [a-z0-9]+ 正则切词并小写化（不做词干化，保持实现简洁）；
  中文无词典，退化为逐字切分（单字即"词"），效果弱于 jieba 分词，
  但零依赖；本项目评测主战场是 BEIR 英文数据（nfcorpus），不受影响。

数据存储：config["storage_path"] 目录下的 bm25/ 子目录
  index.json = 倒排索引（{词: [[文档编号, 出现次数], ...]}）+ 全部文本

【与 dense 的对比维度】检索质量（语义 vs 关键词）、查询延迟（无向量化）、
建库耗时（无嵌入模型调用）、磁盘占用（倒排表 vs 浮点向量矩阵）。
"""

import json
import math
import os
import re
import sys
from collections import Counter

# 本方案在存储根目录下的子目录名，不同方案用不同子目录避免数据互相覆盖
STORE_DIR_NAME = "bm25"

# BM25 标准超参数（面试常考：k1 控制词频饱和度，b 控制文档长度归一化强度）
K1 = 1.5
B = 0.75

# 英文/数字词的正则；中文按单字处理（见模块 docstring）
_EN_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RANGE = ("一", "鿿")

# 已加载的索引缓存：避免每次检索都从磁盘重新读取
_cache = {}


def tokenize(text: str) -> list[str]:
    """把文本切成词列表：英文按连续字母数字切词并小写，中文逐字切分"""
    tokens = []
    for ch in text.lower():
        if _CJK_RANGE[0] <= ch <= _CJK_RANGE[1]:
            tokens.append(ch)               # 中文：单字即一个词（无词典的退化方案）
    tokens += _EN_WORD_RE.findall(text.lower())
    return tokens


def _store_dir(config: dict) -> str:
    """本方案的存储目录（config["storage_path"] 下的子目录）"""
    return os.path.join(config["storage_path"], STORE_DIR_NAME)


def build_index(chunks: list, config: dict) -> None:
    """构建倒排索引：统计每个词的文档频率与词频，连同全部文本落盘

    复杂度：一趟扫描所有文档，O(总词数)；不调用嵌入模型，建库极快。
    """
    texts = [chunk.page_content for chunk in chunks]
    print(f"[bm25] 正在构建倒排索引（{len(texts)} 个片段）...")

    # 1. 逐文档分词，统计词频 tf；postings 按文档编号升序追加，天然有序
    postings = {}   # {词: [[文档编号, 该文档内出现次数], ...]}
    doc_tokens = []
    for doc_id, text in enumerate(texts):
        tf = Counter(tokenize(text))
        doc_tokens.append(tf)
        for term, count in tf.items():
            postings.setdefault(term, []).append([doc_id, count])

    # 2. 计算平均文档长度（BM25 长度归一化用）
    avgdl = sum(sum(tf.values()) for tf in doc_tokens) / max(len(doc_tokens), 1)

    # 3. 落盘：倒排索引 + 文本列表（检索时按文档编号取回原文）
    store_dir = _store_dir(config)
    os.makedirs(store_dir, exist_ok=True)
    index_data = {
        "N": len(texts),
        "avgdl": avgdl,
        "postings": postings,
        "texts": texts,
    }
    with open(os.path.join(store_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)
    _cache.clear()
    print(f"[bm25] 已建立 {len(postings)} 个词的倒排索引，"
          f"平均文档长度 {avgdl:.1f}，落盘到 {store_dir}")


def _load_store(config: dict) -> dict:
    """加载倒排索引（带缓存，避免每次检索都读盘）"""
    store_dir = _store_dir(config)
    if store_dir not in _cache:
        with open(os.path.join(store_dir, "index.json"), encoding="utf-8") as f:
            _cache[store_dir] = json.load(f)
    return _cache[store_dir]


def _idf(df: int, n: int) -> float:
    """逆文档频率：词越稀有（df 越小）权重越高"""
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def _doc_lens(index: dict) -> list[int]:
    """全部文档的长度列表（各词频之和），计算一次后挂在索引字典上复用"""
    if "doc_lens" not in index:
        lens = [0] * index["N"]
        for posting in index["postings"].values():
            for doc_id, tf in posting:
                lens[doc_id] += tf
        index["doc_lens"] = lens
    return index["doc_lens"]


def search(question: str, config: dict) -> list[str]:
    """检索：问题分词 → 查倒排 → BM25 打分 → 返回得分最高的原文片段

    只计算包含问题词的文档（倒排索引的价值），
    检索延迟不含任何嵌入模型调用，通常远快于 dense。
    """
    index = _load_store(config)
    n = index["N"]
    avgdl = index["avgdl"]
    postings = index["postings"]

    # 1. 问题分词后累加各词对候选文档的贡献（候选文档只来自倒排命中）
    doc_lens = _doc_lens(index)   # 文档长度一次性算好，避免循环里逐词重复统计
    scores = {}   # {文档编号: BM25 得分}
    for term in tokenize(question):
        posting = postings.get(term)
        if posting is None:
            continue   # 问题中的词在语料中从未出现，对打分无贡献
        idf = _idf(len(posting), n)
        for doc_id, tf in posting:
            dl = doc_lens[doc_id]
            denom = tf + K1 * (1 - B + B * dl / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1) / denom

    # 2. 按得分降序取前 k 个，返回原文
    k = min(config["n_results"], n)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [index["texts"][doc_id] for doc_id, _ in ranked]


# --- 自检：直接运行本文件，验证建库与检索链路可用 ---
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # 解决 Windows 控制台中文乱码
    # 项目根目录加入搜索路径：直接以文件路径运行本脚本时，rag_demo.py 在上一级目录
    _PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
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
        # 加载 BEIR 块文件（与主程序 BEIR 模式一致）
        chunks_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "beir_chunks.json")
        with open(chunks_file, encoding="utf-8") as f:
            items = json.load(f)
        chunks = [Document(page_content=item["text"], metadata={"chunk": item["id"]})
                  for item in items]
        build_index(chunks, config)

    demo_q = "Do Cholesterol Statin Drugs Cause Breast Cancer?"
    results = search(demo_q, config)
    print(f"\n===== 检索结果（方案：bm25，问题：{demo_q}）=====")
    for i, text in enumerate(results, start=1):
        print(f"\n--- 片段{i}（前150字符）---\n{text[:150]}")
