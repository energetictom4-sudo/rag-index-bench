# -*- coding: utf-8 -*-
"""文本块存储压缩实验（改进路线第 4 项 · 存储延伸的第一站）

背景：阶段三的 PQ 只压缩了向量部分（128 倍），但 RAG 索引的落盘文件里
文本（texts.json）才是存储大头。本实验实测三个数据集块文本在通用压缩下的
压缩比与重复度，回答"文本部分能压多少"，与 PQ 的向量压缩合成完整的
双层存储压缩故事。

用法：
  python experiments/text_compress.py

输出：
  控制台对比表 + experiments/results/text_compress.csv
"""

import csv
import json
import lzma
import os
import sys
import zlib

sys.stdout.reconfigure(encoding="utf-8")

try:
    import zstandard as zstd
except ImportError:
    zstd = None

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.dirname(PROJECT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

DATASETS = ["nfcorpus", "scifact", "fiqa"]
VEC_DIM = 1024      # bge-m3 输出维度（与 indexes/dense.py 一致）
PQ_RATIO = 128      # PQ M=32 的实测压缩比（README 阶段三 PQ 小节）


def chunks_file(ds: str) -> str:
    """当前数据集的块文件路径（与 eval_retrieval.py 的命名约定一致）"""
    if ds == "nfcorpus":
        return os.path.join(DATA_ROOT, "beir_chunks.json")
    return os.path.join(DATA_ROOT, f"beir_chunks_{ds}.json")


def mb(n: int) -> float:
    return n / (1024 * 1024)


def analyze(ds: str) -> dict:
    with open(chunks_file(ds), encoding="utf-8") as f:
        items = json.load(f)
    texts = [it["text"] for it in items]
    n = len(texts)

    # 落盘形态与 indexes/dense.py 的 texts.json 一致：UTF-8 的 JSON 文本数组
    raw = json.dumps(texts, ensure_ascii=False).encode("utf-8")

    # 重复度：完全重复的块文本占比（去重是压缩之前零成本的存储优化）
    uniq = set(texts)
    dup_ratio = 1.0 - len(uniq) / n

    # 通用无损压缩（对落盘字节流直接压缩，与真实存储方案一致）
    gz = zlib.compress(raw, level=9)
    xz = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    zs = zstd.ZstdCompressor(level=19).compress(raw) if zstd else None

    # 向量部分的对照（faiss_dense 的 float32 矩阵，L2 归一化后仍为 float32）
    vec_raw = n * VEC_DIM * 4

    return {
        "数据集": ds, "块数": n,
        "文本原始MB": round(mb(len(raw)), 2),
        "gzipMB": round(mb(len(gz)), 2),
        "gzip倍数": round(len(raw) / len(gz), 1),
        "zstdMB": round(mb(len(zs)), 2) if zs else None,
        "zstd倍数": round(len(raw) / len(zs), 1) if zs else None,
        "xzMB": round(mb(len(xz)), 2),
        "xz倍数": round(len(raw) / len(xz), 1),
        "重复块占比": f"{dup_ratio:.1%}",
        "向量原始MB": round(mb(vec_raw), 2),
        "向量PQ后MB": round(mb(vec_raw) / PQ_RATIO, 2),
        "PQ倍数": PQ_RATIO,
    }


def main() -> None:
    print("=" * 70)
    print("文本块存储压缩实验：三数据集 texts.json 的无损压缩比与重复度")
    print("=" * 70)
    rows = [analyze(ds) for ds in DATASETS]

    header = ("数据集", "块数", "文本原始MB", "gzip倍数", "zstd倍数", "xz倍数",
              "重复块占比", "向量原始MB", "向量PQ后MB")
    print()
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for r in rows:
        zs = f"{r['zstd倍数']}x" if r["zstd倍数"] else "未安装"
        print(f"{r['数据集']} | {r['块数']} | {r['文本原始MB']} | {r['gzip倍数']}x "
              f"| {zs} | {r['xz倍数']}x | {r['重复块占比']} "
              f"| {r['向量原始MB']} | {r['向量PQ后MB']}")

    # 落盘 csv
    csv_path = os.path.join(RESULTS_DIR, "text_compress.csv")
    fields = ["数据集", "块数", "文本原始MB", "gzipMB", "gzip倍数", "zstdMB",
              "zstd倍数", "xzMB", "xz倍数", "重复块占比", "向量原始MB", "向量PQ后MB"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\n已写入 {csv_path}")

    # 结论速览
    print("\n结论速览：")
    for r in rows:
        vec = r["向量原始MB"]
        text = r["文本原始MB"]
        share = text / (text + vec)
        print(f"  {r['数据集']}：文本占索引落盘的 {share:.0%}（文本 {text}MB vs "
              f"向量 {vec}MB），gzip 可压 {r['gzip倍数']}x，重复块 {r['重复块占比']}")


if __name__ == "__main__":
    main()
