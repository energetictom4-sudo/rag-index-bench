# -*- coding: utf-8 -*-
"""
实验第 3 步：错误分析 —— 文档级与块级差异、零召回题复查、A/B 差异归因

读取第 1/2 步产出的逐题结果 JSON（纯离线分析，不调用 Ollama），输出：
  1. 三口径总览：方案 A（整文档向量化）vs 方案 B（块向量池化）vs 块级检索对照
  2. A/B 对比结论：差距数字 + "池化是否可行"的判断
  3. 零召回题复查：列出 Recall@100=0 的题（题面、相关文档数、相关文档标题），供人工排查
  4. A/B 差异归因：按"相关文档平均块数（文档长度代理）"分桶，
     观察长文档/短文档在 A、B 方案下的表现差异，判断块均值是否损失全局语义
  5. 与块级口径的差异说明（实验方案表0 四维对照的实测数字）

用法：
  python experiments/analyze.py

输出：
  reports/错误分析报告.txt
  reports/实验总结报告.txt   （含第 4 步决策建议，正式代码此时仍不动）
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

from common import (BASELINES, CHUNKS_FILE, K_LIST, NDCG_K, REPORTS_DIR, RESULTS_DIR,
                    load_corpus_fulltexts, load_qrels)

REPORT_FILE = os.path.join(REPORTS_DIR, "错误分析报告.txt")
SUMMARY_FILE = os.path.join(REPORTS_DIR, "实验总结报告.txt")
FULL_RESULT = os.path.join(RESULTS_DIR, "full_doc_per_query.json")     # 方案 A
POOL_RESULT = os.path.join(RESULTS_DIR, "pool_doc_per_query.json")     # 方案 B
BLOCK_RESULT = os.path.join(RESULTS_DIR, "block_level_per_query.json") # 块级对照


def load_result(path):
    """读取逐题结果 JSON，返回 {qid: 结果dict}"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {r["qid"]: r for r in items}


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    # ============ 0. 读取三份逐题结果 ============
    full = load_result(FULL_RESULT)
    pool = load_result(POOL_RESULT)
    block = load_result(BLOCK_RESULT)
    common_qids = [qid for qid in full if qid in pool]
    log(f"已加载：方案A {len(full)} 题 / 方案B {len(pool)} 题 / 块级对照 {len(block)} 题，"
        f"A/B 共有 {len(common_qids)} 题")

    # ============ 1. 三口径总览 ============
    log("")
    log("=" * 70)
    log("【1】三口径总览（全部全量 qrels 不截断；块级对照为 Top-50 块映射文档）")
    log("=" * 70)
    header = f"  {'指标':<12}" + "".join(f"{name:>14}" for name in
                                          ["方案A 全文", "方案B 池化", "块级对照(50块)"])
    log(header)

    def avg(rs, key):
        return sum(r[key] for r in rs) / len(rs)

    def avg_recall(rs, k):
        return sum(r["recall"][f"recall@{k}"] for r in rs) / len(rs)

    fa, pb = [full[q] for q in common_qids], [pool[q] for q in common_qids]
    bl = [block[q] for q in common_qids]
    for k in K_LIST:
        log(f"  {'Recall@' + str(k):<12}" +
            f"{avg_recall(fa, k):>14.4f}"
            f"{avg_recall(pb, k):>14.4f}"
            f"{avg_recall(bl, k):>14.4f}")
    log(f"  {'NDCG@' + str(NDCG_K):<12}{avg(fa, 'ndcg'):>14.4f}{avg(pb, 'ndcg'):>14.4f}"
        f"{avg(bl, 'ndcg'):>14.4f}")

    # 零召回统计
    zf = sum(1 for q in common_qids if not full[q]["hit_docs"])
    zp = sum(1 for q in common_qids if not pool[q]["hit_docs"])
    zb = sum(1 for q in common_qids if not block[q]["hit_docs"])
    log(f"  {'零召回题数':<12}{zf:>14}{zp:>14}{zb:>14}   （共 {len(common_qids)} 题）")

    # ============ 2. A/B 对比结论 ============
    ndcg_a = avg(fa, "ndcg")
    ndcg_b = avg(pb, "ndcg")
    log("")
    log("=" * 70)
    log("【2】A/B 对比结论：池化是否可行？")
    log("=" * 70)
    log(f"  方案A（全文向量化）NDCG@{NDCG_K} = {ndcg_a:.4f}")
    log(f"  方案B（块向量池化）NDCG@{NDCG_K} = {ndcg_b:.4f}")
    diff = ndcg_a - ndcg_b
    log(f"  差距 A−B = {diff:+.4f}（相对 {(diff / ndcg_a * 100) if ndcg_a else 0:+.1f}%）")
    a_better = sum(1 for q in common_qids if full[q]["ndcg"] > pool[q]["ndcg"] + 1e-9)
    b_better = sum(1 for q in common_qids if pool[q]["ndcg"] > full[q]["ndcg"] + 1e-9)
    tie = len(common_qids) - a_better - b_better
    log(f"  逐题胜负：A 胜 {a_better} 题 / B 胜 {b_better} 题 / 平 {tie} 题")
    # 命中文档重叠度：同一题 A、B 检索命中的相关文档集合一致性
    overlap = np.mean([
        len({d for d, _ in full[q]["hit_docs"]} & {d for d, _ in pool[q]["hit_docs"]})
        / max(len(full[q]["hit_docs"]), 1) for q in common_qids])
    log(f"  A/B 命中文档平均重叠度（以A为分母）：{overlap:.1%}")
    if abs(diff) < 0.01:
        log("  结论：A≈B（差距<0.01），块向量平均池化可行——")
        log("        无需重新向量化，直接复用现有块向量即可构建文档级索引，零成本。")
    elif diff > 0:
        log(f"  结论：A 明显更好（差距 {diff:.3f}），块级平均损失了文档的全局语义——")
        log("        建议正式方案采用整文档重新向量化（3633 次请求，几分钟建库）。")
    else:
        log(f"  结论：B 反而更好（{diff:+.3f}），池化不仅零成本且效果更优，直接采用池化方案。")
    # 与基线对照（方案A为准）
    log("")
    log("  对标 BEIR 官方基线（以方案 A 为准）：")
    for name, v in BASELINES:
        log(f"    {name:<16} {v:.3f}")
    log(f"    本实验方案A      {ndcg_a:.4f}")

    # ============ 3. 零召回题复查 ============
    log("")
    log("=" * 70)
    log("【3】零召回题复查（Top-100 一个相关文档都没召回的题）")
    log("=" * 70)
    zero_qids = [q for q in common_qids if not full[q]["hit_docs"]]
    # 相关文档标题（用于人工判断零召回是否"情有可原"：题面与文档确实难匹配）
    doc_title = {}
    for did, text in zip(*load_corpus_fulltexts()):
        doc_title[did] = text.split("\n")[0][:80]
    qrels = load_qrels()
    log(f"方案A 零召回 {len(zero_qids)} 题（另有方案B 独有零召回 "
        f"{sum(1 for q in common_qids if not pool[q]['hit_docs'] and full[q]['hit_docs'])} 题）")
    log("（评分1/2 与全部相关文档的标题列表如下，判断属模型弱还是题难）")
    for qid in zero_qids[:15]:   # 最多细列 15 题，其余只列题号
        rels = sorted(qrels[qid].items(), key=lambda kv: -kv[1])
        log(f"\n  [{qid}] {full[qid]['question'][:100]}")
        log(f"    相关文档 {full[qid]['rel_total']} 篇（等级降序）：")
        for did, sc in rels[:5]:
            log(f"      评分{sc}：{doc_title.get(did, did)}")
        if len(rels) > 5:
            log(f"      ... 其余 {len(rels) - 5} 篇略")
    if len(zero_qids) > 15:
        log(f"\n  （其余 {len(zero_qids) - 15} 题题号：{', '.join(zero_qids[15:])}）")

    # ============ 4. A/B 差异归因：文档长度（块数）视角 ============
    log("")
    log("=" * 70)
    log("【4】A/B 差异归因：相关文档长度（块数）分桶")
    log("=" * 70)
    log('若池化损失全局语义，长文档（块多、均值更"稀释"）在方案B下的劣势应更明显。')
    # 每篇文档的块数
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunk_items = json.load(f)
    doc_nblocks = defaultdict(int)
    for item in chunk_items:
        doc_nblocks[item["doc_id"]] += 1

    # 每道题的"相关文档平均块数"与 A−B 的 NDCG 差
    rows = []   # (平均块数, A的NDCG−B的NDCG, A的NDCG, B的NDCG, 题号)
    for qid in common_qids:
        rel_docs = qrels[qid]
        avg_blk = np.mean([doc_nblocks.get(d, 0) for d in rel_docs])
        rows.append((avg_blk, full[qid]["ndcg"] - pool[qid]["ndcg"],
                     full[qid]["ndcg"], pool[qid]["ndcg"], qid))
    rows.sort()
    n_bucket = 4
    bucket_size = len(rows) // n_bucket
    log(f"  {'相关文档平均块数分桶':<24}{'题数':>6}{'A的NDCG':>10}{'B的NDCG':>10}{'差(A−B)':>10}")
    for b in range(n_bucket):
        part = rows[b * bucket_size:(b + 1) * bucket_size]
        lo = part[0][0]
        hi = part[-1][0]
        log(f"  {f'{lo:.1f} ~ {hi:.1f} 块':<24}{len(part):>6}"
            f"{np.mean([r[2] for r in part]):>10.4f}{np.mean([r[3] for r in part]):>10.4f}"
            f"{np.mean([r[1] for r in part]):>+10.4f}")
    # 相关系数：平均块数 vs A−B 差（正相关=长文档题A优势更大→池化损失全局语义）
    blk_arr = np.array([r[0] for r in rows])
    diff_arr = np.array([r[1] for r in rows])
    corr = np.corrcoef(blk_arr, diff_arr)[0, 1]
    log(f"  相关文档平均块数 与 A−B差的相关系数 = {corr:+.3f}")
    if corr > 0.2:
        log("  → 正相关：相关文档越长，方案A优势越大，印证池化损失全局语义（尤其长文档）。")
    elif corr < -0.2:
        log("  → 负相关：相关文档越长，方案B反而越好，说明池化对长文档有利（少见，值得细看）。")
    else:
        log("  → 无明显相关：A/B 差距与文档长度关系不大，差距更可能来自其他因素。")

    # ============ 5. 与块级口径的差异说明 ============
    log("")
    log("=" * 70)
    log("【5】文档级 vs 块级口径差异（实验方案表0 四维对照的实测数字）")
    log("=" * 70)
    n_docs = len(doc_nblocks)
    log(f"  索引单位：块级 17071 个块 → 文档级 {n_docs} 篇文档（方案A/B 均为文档级）")
    log(f"  检索返回：块级 Top-50 块（去重后约覆盖 10 篇左右文档）→ 文档级 Top-100 篇文档")
    log(f"  评测分母：块级旧口径按截断后的相关块映射文档 → 本次全量 qrels "
        f"（{sum(len(v) for v in qrels.values())} 条，不截断）")
    log(f"  可比对象：本次数字可直接对照 BEIR 论文官方基线；块级旧口径无法直接对照")
    # 实例：随机挑一题对比两种口径下"同样检索深度"的覆盖差异
    demo_qid = common_qids[0]
    blk_covered = len({d for d, _ in block[demo_qid]["hit_docs"]})
    full_hit = full[demo_qid]["recall"]["recall@100"]
    log(f"  实例[{demo_qid}]：块级 Top-50 块命中 {blk_covered} 篇相关文档；"
        f"方案A Top-100 文档召回率 {full_hit:.1%}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\n错误分析报告已保存到 {REPORT_FILE}")

    # ============ 6. 实验总结报告（含第 4 步决策建议） ============
    summary = []
    summary.append("=" * 70)
    summary.append("文档级检索实验总结报告（实验方案第 4 步：决策建议）")
    summary.append("=" * 70)
    summary.append("")
    summary.append("一、实验结果一览（nfcorpus，全量 qrels，官方口径）")
    summary.append("")
    for k in K_LIST:
        summary.append(f"  Recall@{k:<4}  方案A全文={avg_recall(fa, k):.4f}   "
                       f"方案B池化={avg_recall(pb, k):.4f}")
    summary.append(f"  NDCG@{NDCG_K}   方案A全文={ndcg_a:.4f}   方案B池化={ndcg_b:.4f}")
    summary.append("")
    summary.append("  BEIR 官方基线对照：BGE-large-v1.5=0.380  SPLADE++=0.347  "
                   "BM25=0.325  TAS-B/GenQ=0.319  SBERT=0.272")
    if ndcg_a >= 0.325:
        summary.append(f"  结论：方案A NDCG@{NDCG_K}={ndcg_a:.4f} 达到 BM25 基线，裸模板无实质损伤")
    elif ndcg_a >= 0.272:
        summary.append(f"  结论：方案A NDCG@{NDCG_K}={ndcg_a:.4f} 达到 SBERT 基线以上，"
                       f"无实质损伤，但距 BM25 仍有差距")
    else:
        summary.append(f"  结论：方案A NDCG@{NDCG_K}={ndcg_a:.4f} 低于 SBERT 基线，"
                       f"嵌入配置存在实质损伤，需排查")
    summary.append("")
    summary.append("二、决策建议（正式代码这一步仍不动，确认后再动手）")
    summary.append("")
    if abs(ndcg_a - ndcg_b) < 0.01:
        summary.append(f"  A、B 差距 {abs(ndcg_a - ndcg_b):.3f} < 0.01：推荐采用【方案B：块向量平均池化】。")
        summary.append("  理由：效果相当且零向量化成本，复用现有块向量即可构建文档级索引，")
        summary.append("        建库耗时从分钟级降到秒级，且不用维护第二份向量缓存。")
    elif ndcg_a > ndcg_b:
        summary.append(f"  A 明显优于 B（差距 {ndcg_a - ndcg_b:.3f}）：推荐采用【方案A：整文档重新向量化】。")
        summary.append("  理由：块级平均池化损失了文档全局语义，牺牲的效果不值得省一次向量化；")
        summary.append("        3633 次向量化请求几分钟跑完，代价可接受。")
    else:
        summary.append(f"  B 优于 A（{ndcg_b - ndcg_a:+.3f}）：推荐采用【方案B：块向量平均池化】。")
        summary.append("  理由：零成本且效果更好，没有理由重新向量化。")
    summary.append("")
    summary.append("三、正式落地步骤（确认方案后执行，此刻不动正式代码）")
    summary.append("")
    summary.append("  1. 在 indexes/ 新建 doc_dense.py（接口约定见 indexes/README.md）：")
    summary.append("     - 方案A：corpus 全文向量化 → IndexFlatIP → 落盘子目录 faiss_doc_dense/")
    summary.append("     - 方案B：读 faiss_dense/ 块向量按 doc_id 池化 → 落盘子目录 faiss_doc_pool/")
    summary.append("  2. \"先搜文档再取块\"的取块逻辑一并实现：文档级检索命中 Top-K 文档后，")
    summary.append("     按 prepare_beir.py 的 doc_to_chunks 映射取该文档全部块、按原顺序拼成上下文，")
    summary.append("     喂给 RAG 生成环节（rag_demo.py 的 generate_answer 无需改动）。")
    summary.append("  3. 在主程序 rag_demo.py 配置区把 INDEX_METHOD 改为新方案文件名，")
    summary.append("     并把 N_RESULTS 调到文档级档位（如 10）即可上线。")
    summary.append("  4. 上线后用 eval_retrieval.py 复跑一遍验证正式方案的指标与实验一致。")
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    log(f"实验总结报告已保存到 {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
