# -*- coding: utf-8 -*-
"""
纯检索延迟微基准：对比 dense（IndexFlatIP 暴力检索）与 ivf（IVF 近似检索）、
hnsw（HNSW 图索引近似检索）的纯 search() 耗时

为什么单独测纯检索：
  eval_retrieval.py 测的是端到端延迟（含问题向量化约 250ms），暴力扫描与近似索引的差距
  只有毫秒级，会被向量化开销完全淹没。本脚本预热后只测 index.search() 本身，
  才能看出索引结构带来的真实加速（阶段三的核心测量工具）。

用法：
  python experiments/bench_retrieval.py
      前置：faiss_dense/、faiss_ivf/、faiss_hnsw/ 三个索引均已建好（先分别跑过对应方案的评测）
输出：
  dense 暴力检索参照 + ivf 各 nprobe 档位 + hnsw 各 ef_search 档位的平均/中位/最小延迟
  （毫秒）与相对暴力检索的加速比。

说明：
  查询向量用固定随机种子生成并 L2 归一化（与真实查询分布一致；纯检索耗时与向量内容无关）。
  后续可扩展规模扫描（把向量复制放大到 10 万~100 万块）寻找"暴力检索扛不住"的交叉点。
"""

import os
import sys
import time

import faiss
import numpy as np

# 解决 Windows 控制台中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

# ================ 参数配置区 ================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)   # 与 rag_demo._DATA_ROOT / experiments/common.py 一致
DENSE_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_dense", "index.faiss")
IVF_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_ivf", "index.faiss")
HNSW_INDEX_FILE = os.path.join(DATA_ROOT, "vector_store", "faiss_hnsw", "index.faiss")

N_QUERIES = 1000      # 随机查询向量数量（逐个计时取统计）
WARMUP = 10           # 预热次数（不计时）
TOPK = 50             # 每次检索返回的片段数（与评测脚本 QUERY_TOP_K 一致）
SEED = 42             # 随机种子（结果可复现）

NPROBE_LIST = (1, 4, 8, 16, 32, 64, 128, 256)   # ivf 扫描的 nprobe 档位
EF_SEARCH_LIST = (4, 8, 16, 32, 64, 128, 256)   # hnsw 扫描的 ef_search 档位（与 nprobe 档位对齐）

# ======================== 以下为代码逻辑，无需改动 ========================


def load_index(path: str) -> faiss.Index:
    """加载 FAISS 索引；文件不存在时给出明确提示"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引文件不存在：{path}\n请先运行对应方案的评测建库（python eval_retrieval.py）")
    return faiss.read_index(path)


def make_queries(d: int, n: int, seed: int) -> np.ndarray:
    """生成 n 个归一化的随机查询向量（维度 d，形状 (n, d)）"""
    rng = np.random.default_rng(seed)
    queries = rng.standard_normal((n, d)).astype(np.float32)
    faiss.normalize_L2(queries)
    return queries


def bench_search(index: faiss.Index, queries: np.ndarray, topk: int) -> dict:
    """逐个查询计时（预热后），返回延迟统计（毫秒）"""
    times = []
    for q in queries[:WARMUP]:      # 预热：触发惰性初始化、预热缓存
        index.search(q.reshape(1, -1), topk)
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), topk)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "avg": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    dense_index = load_index(DENSE_INDEX_FILE)
    ivf_index = load_index(IVF_INDEX_FILE)
    hnsw_index = load_index(HNSW_INDEX_FILE)
    d = dense_index.d
    n_total = dense_index.ntotal
    print(f"语料规模：{n_total} 个块向量（维度 {d}），查询数 {N_QUERIES}，Top-K={TOPK}")
    print(f"随机种子：{SEED}（查询向量固定，结果可复现）\n")

    queries = make_queries(d, N_QUERIES, SEED)

    # 1. dense 暴力检索参照
    stats = bench_search(dense_index, queries, TOPK)
    dense_avg = stats["avg"]
    print("=" * 64)
    print("【纯检索延迟对比】（不含问题向量化，单位：毫秒）")
    print("=" * 64)
    print(f"{'方案':<24}{'平均':>10}{'中位':>10}{'最小':>10}{'最大':>10}{'加速比':>9}")
    print(f"{'dense (IndexFlatIP)':<24}{stats['avg']:>10.3f}{stats['median']:>10.3f}"
          f"{stats['min']:>10.3f}{stats['max']:>10.3f}{'1.00x（参照）':>9}")

    # 2. ivf 各 nprobe 档位
    for nprobe in NPROBE_LIST:
        ivf_index.nprobe = nprobe
        stats = bench_search(ivf_index, queries, TOPK)
        speedup = dense_avg / stats["avg"]
        print(f"{f'ivf (nprobe={nprobe})':<24}{stats['avg']:>10.3f}{stats['median']:>10.3f}"
              f"{stats['min']:>10.3f}{stats['max']:>10.3f}{f'{speedup:.2f}x':>9}")

    # 3. hnsw 各 ef_search 档位
    for ef in EF_SEARCH_LIST:
        hnsw_index.hnsw.efSearch = ef
        stats = bench_search(hnsw_index, queries, TOPK)
        speedup = dense_avg / stats["avg"]
        print(f"{f'hnsw (ef={ef})':<24}{stats['avg']:>10.3f}{stats['median']:>10.3f}"
              f"{stats['min']:>10.3f}{stats['max']:>10.3f}{f'{speedup:.2f}x':>9}")

    print("\n说明：加速比 = dense 平均延迟 / 对应方案平均延迟，>1 表示比暴力检索快。")
    print("档位参数越小延迟越低，但召回损失越大；两者关系见 eval_retrieval.py 的召回评测，")
    print("联合画召回率-延迟曲线（阶段三核心产出，见 ann_curve.py）。")


if __name__ == "__main__":
    main()
