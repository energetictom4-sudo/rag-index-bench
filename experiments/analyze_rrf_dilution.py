# -*- coding: utf-8 -*-
"""RRF 稀释的题目分布拆解（改进路线第 5 项 · 机理验证的第一部分）

问题：hybrid（RRF 融合）在 nfcorpus 上浅档位低于 dense（"稀释"），深档位反超。
本脚本把"稀释"拆到题目粒度：哪些题被 hybrid 稀释、哪些被增强，两类题的查询
有什么特征（词数、数字、长度等）——从"报数字"走向"解释原因"。

用法：
  python experiments/analyze_rrf_dilution.py

输入：评测报告_dense_beir.txt 与 评测报告_hybrid_beir.txt（nfcorpus，同为 119 题口径）
输出：experiments/results/rrf_dilution.csv + 控制台分布与典型题例
"""

import csv
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSE_REPORT = os.path.join(PROJECT_DIR, "评测报告_dense_beir.txt")
HYBRID_REPORT = os.path.join(PROJECT_DIR, "评测报告_hybrid_beir.txt")
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiments", "results")


def parse_per_query(path: str) -> dict:
    """解析报告逐题日志，返回 {题id: (问题文本, 召回率@50百分比)}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    cur_id, cur_text, cur_recall = None, None, None
    for line in lines:
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
        if m and "类别" not in line and "召回率" not in line:
            # 新题开始（[题号] 后面是问题文本；"类别"/"召回率"行不匹配此分支）
            if cur_id is not None and cur_recall is not None:
                out[cur_id] = (cur_text, cur_recall)
            cur_id, cur_text, cur_recall = m.group(1), m.group(2).strip(), None
        m = re.search(r"召回率@50\s*=\s*\d+/\d+\s*=\s*([\d.]+)%", line)
        if m and cur_id is not None:
            cur_recall = float(m.group(1))
    if cur_id is not None and cur_recall is not None:
        out[cur_id] = (cur_text, cur_recall)
    return out


def query_features(text: str) -> dict:
    """查询文本的简单特征：词数、数字个数、长度、是否含括号"""
    words = text.split()
    return {
        "词数": len(words),
        "字符数": len(text),
        "数字个数": sum(1 for ch in text if ch.isdigit()),
        "含括号": 1 if ("(" in text or ")" in text) else 0,
    }


def main() -> None:
    print("=" * 70)
    print("RRF 稀释题目分布拆解（nfcorpus，dense vs hybrid，Recall@50）")
    print("=" * 70)
    dense = parse_per_query(DENSE_REPORT)
    hybrid = parse_per_query(HYBRID_REPORT)
    common = sorted(set(dense) & set(hybrid),
                    key=lambda x: int(x.split("-")[-1]))
    print(f"两报告共同题数：{len(common)}（dense 报告 {len(dense)} 题 / hybrid 报告 "
          f"{len(hybrid)} 题）")

    rows = []
    for qid in common:
        text, dr = dense[qid]
        hr = hybrid[qid][1]
        rows.append((qid, text, dr, hr, hr - dr))
    rows.sort(key=lambda r: r[4])          # 按差异升序：最稀释在前

    n_dil = sum(1 for r in rows if r[4] < 0)
    n_gain = sum(1 for r in rows if r[4] > 0)
    n_same = len(rows) - n_dil - n_gain
    print(f"\n差异分布：被稀释（hybrid 更低）{n_dil} 题 / 持平 {n_same} 题 / "
          f"被增强 {n_gain} 题")
    print(f"平均差异：{sum(r[4] for r in rows) / len(rows):+.2f}pp")

    # 典型题例
    print("\n【被稀释最严重的 8 题】（hybrid 显著低于 dense）")
    for qid, text, dr, hr, diff in rows[:8]:
        print(f"  [{qid}] {diff:+.1f}pp（dense {dr:.1f}% → hybrid {hr:.1f}%）"
              f"  查询：{text[:70]}")
    print("\n【被增强最明显的 8 题】（hybrid 显著高于 dense）")
    for qid, text, dr, hr, diff in rows[-8:]:
        print(f"  [{qid}] {diff:+.1f}pp（dense {dr:.1f}% → hybrid {hr:.1f}%）"
              f"  查询：{text[:70]}")

    # 特征对比：稀释题 vs 增强题
    dil = [r for r in rows if r[4] < 0]
    gain = [r for r in rows if r[4] > 0]
    print("\n【查询特征对比】")
    for feat in ("词数", "字符数", "数字个数", "含括号"):
        dv = sum(query_features(r[1])[feat] for r in dil) / max(len(dil), 1)
        gv = sum(query_features(r[1])[feat] for r in gain) / max(len(gain), 1)
        print(f"  {feat}：稀释题平均 {dv:.1f}  vs  增强题平均 {gv:.1f}")

    # 落盘 csv
    csv_path = os.path.join(RESULTS_DIR, "rrf_dilution.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["题id", "查询", "dense召回50", "hybrid召回50", "差异pp",
                    "词数", "字符数", "数字个数", "含括号"])
        for qid, text, dr, hr, diff in rows:
            feats = query_features(text)
            w.writerow([qid, text, dr, hr, round(diff, 2), feats["词数"],
                        feats["字符数"], feats["数字个数"], feats["含括号"]])
    print(f"\n已写入 {csv_path}")


if __name__ == "__main__":
    main()
