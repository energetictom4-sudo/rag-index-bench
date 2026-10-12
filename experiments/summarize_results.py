# -*- coding: utf-8 -*-
"""评测结果产物化脚本：把各方案评测报告解析为统一数据表

用法：
  python experiments/summarize_results.py        # 扫描项目根目录的评测报告_*.txt

输出：
  1. experiments/results/summary.csv —— 一行一个 (方案, 数据集)，列含 Recall/Hit 四档位、
     平均查询延迟、建库耗时、索引文件大小
  2. 控制台打印三数据集的 markdown 对比表（与 README 表格对照核对用）

设计意图（改进路线第 3 项）：对比表由本脚本从报告文件生成，杜绝手抄错误
（2026-10-11 命中率列曾抄串行过一次）。README 表格更新流程：跑评测 → 跑本脚本 →
用输出的表替换 README 对应小节。
"""

import csv
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")

# 评测报告文件名格式：评测报告_{方法}_beir.txt（nfcorpus）/
#                   评测报告_{方法}_{数据集}_beir.txt（其他数据集）
REPORT_PATTERN = re.compile(r"评测报告_(.+?)(?:_(scifact|fiqa))?_beir\.txt")
# 方案名可能与数据集名混淆：如 评测报告_dense_scifact_beir.txt 中方案=dense 数据集=scifact。
# 用已知方案名集合反向消解（先按后缀拆，后缀不匹配已知数据集时整体视为方案名）。
KNOWN_DATASETS = {"scifact", "fiqa"}
KNOWN_METHODS = {"dense", "doc_dense", "bm25", "hybrid", "parent_doc",
                 "ivf", "hnsw", "pq"}


def parse_report(path: str) -> dict:
    """解析一份评测报告，返回指标字典；解析不到的行置 None"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {k: None for k in (
        "recall5", "recall10", "recall20", "recall50",
        "hit5", "hit10", "hit20", "hit50",
        "latency_ms", "build_time_s", "index_mb")}

    # 召回率摘要块：Recall@5=xx%  ...  / Hit@5=xx%  ...
    for name in ("recall", "hit"):
        for k in (5, 10, 20, 50):
            m = re.search(rf"\b{name.title()}@{k}=([\d.]+)%", text)
            if m:
                out[f"{name}{k}"] = float(m.group(1))
    # Hit@k 摘要行是 2026-10-11 才加入报告格式的；老报告从逐档位汇总行
    # "Top-k（共N题）：...题目命中率=xx%" 回退解析
    for k in (5, 10, 20, 50):
        if out[f"hit{k}"] is None:
            m = re.search(rf"Top-{k}（共\d+题）：.*?题目命中率=([\d.]+)%", text)
            if m:
                out[f"hit{k}"] = float(m.group(1))
    # 查询延迟：总体：平均 xx 毫秒
    m = re.search(r"总体：平均 ([\d.]+) 毫秒", text)
    if m:
        out["latency_ms"] = float(m.group(1))
    # 建库耗时与索引文件大小（--skip-build 的报告无资源占用块，置 None 即可）
    m = re.search(r"建库耗时：([\d.]+) 秒", text)
    if m:
        out["build_time_s"] = float(m.group(1))
    m = re.search(r"索引文件大小：本次建库写入 \d+ 个文件，共 ([\d.]+) MB", text)
    if m:
        out["index_mb"] = float(m.group(1))
    return out


def scan_reports() -> list[tuple]:
    """扫描报告文件，返回 [(方案名, 数据集名, 指标dict, 文件路径)]，按 数据集/方案 排序"""
    rows = []
    for fname in os.listdir(PROJECT_DIR):
        if not fname.startswith("评测报告_") or not fname.endswith(".txt"):
            continue
        stem = fname[len("评测报告_"):-len("_beir.txt")]
        method, dataset = stem, "nfcorpus"
        for ds in KNOWN_DATASETS:
            if stem.endswith(f"_{ds}"):
                method, dataset = stem[:-len(f"_{ds}")], ds
                break
        if method not in KNOWN_METHODS:
            print(f"  [跳过] 无法识别方案的报告：{fname}")
            continue
        rows.append((method, dataset, parse_report(os.path.join(PROJECT_DIR, fname)), fname))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def main() -> None:
    print("=" * 70)
    print("评测结果产物化：解析各方案报告 → summary.csv + markdown 对比表")
    print("=" * 70)
    rows = scan_reports()
    if not rows:
        print("未找到任何评测报告")
        return

    # 1. 写 csv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    fields = ["方案", "数据集", "Recall@5", "Recall@10", "Recall@20", "Recall@50",
              "Hit@5", "Hit@10", "Hit@20", "Hit@50",
              "平均延迟ms", "建库耗时s", "索引文件大小MB"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for method, dataset, d, _ in rows:
            w.writerow([method, dataset, d["recall5"], d["recall10"], d["recall20"],
                        d["recall50"], d["hit5"], d["hit10"], d["hit20"], d["hit50"],
                        d["latency_ms"], d["build_time_s"], d["index_mb"]])
    print(f"\n已写入 {csv_path}（{len(rows)} 行）\n")

    # 2. 打印 markdown 对比表（每数据集一张，含 Recall 与 Hit@50）
    for ds in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == ds]
        print(f"### {ds}（{len(sub)} 方案）")
        print("| 方案 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Hit@50 |")
        print("|---|---:|---:|---:|---:|---:|")
        for method, _, d, _ in sub:
            def fmt(v):
                return f"{v:.2f}%" if v is not None else "—"
            print(f"| {method} | {fmt(d['recall5'])} | {fmt(d['recall10'])} | "
                  f"{fmt(d['recall20'])} | {fmt(d['recall50'])} | {fmt(d['hit50'])} |")
        print()


if __name__ == "__main__":
    main()
