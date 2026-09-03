# -*- coding: utf-8 -*-
"""
检索评测脚本：对 rag_demo.py 中指定的索引方案进行多指标评测

评测指标：
  1. 召回率 / 精确率 / F1：逐题检索，通过文本内容反查块编号，与题库"相关块编号"比对
  2. 查询延迟：每道题预热1次后重复检索取平均（毫秒，含问题向量化的耗时）
  3. 内存占用：建库前后评测进程的内存增量（需安装 psutil，可选）
  4. 磁盘占用：建库前后 chroma_db 目录的大小增量
  5. 建库耗时：构建索引的总时间

设计说明：
  - 完全复用主程序 rag_demo.py 的配置与索引方案加载逻辑：主程序里改 INDEX_METHOD
    即切换评测对象，不存在两套配置不一致的问题
  - 索引方案接口 search() 只返回文本，本脚本通过"文本反查块编号"与题库标注比对，
    因此既能评稠密向量索引，也能评父文档索引等返回大段文本的方案
  - 默认每次重建索引，保证评测结果可复现；--skip-build 可跳过建库
  - 支持两种数据源：DATA_MODE="pdf" 用主程序的PDF与《测试题库.json》；
    DATA_MODE="beir" 用 prepare_beir.py 生成的BEIR数据（beir_chunks.json + beir_题库.json）

用法：
  python eval_retrieval.py              # 重建索引后评测（推荐）
  python eval_retrieval.py --skip-build # 跳过建库，直接评测现有索引

指标说明（对每道有标注的题）：
  召回率 = 检索结果中命中标注块的数量 / 该题标注块总数
  精确率 = 检索结果中命中标注块的数量 / 检索返回片段数(K)
  题目命中率 = 至少召回一个标注块的题目占比
  拒答类题目（标注块为空）不参与指标计算，仅展示检索情况供人工参考

依赖：psutil 可选（未安装则跳过内存指标，安装命令 pip install psutil）
"""

import json
import os
import sys
import time
from datetime import datetime

# 解决 Windows 控制台中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

try:
    import psutil
except ImportError:
    psutil = None

# 复用主程序 rag_demo.py 的全部配置（索引方案名、嵌入模型、切分参数、PDF路径等）
import rag_demo
from langchain_core.documents import Document   # BEIR模式包装块文本用

# ================ 评测专属参数配置区（主程序的参数在 rag_demo.py 里改） ================

# --- 数据源参数 ---
DATA_MODE = "beir"     # "pdf" = 用主程序配置的PDF与《测试题库.json》（原有流程）
                      # "beir" = 用 prepare_beir.py 生成的BEIR数据（块文件+块级题库）
# BEIR数据在项目目录的上一级（与 rag_demo._DATA_ROOT / experiments/common.py 的 DATA_ROOT
# 一致），用绝对路径保证从任何目录运行本脚本都能找到数据
_DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEIR_CHUNKS_FILE = os.path.join(_DATA_ROOT, "beir_chunks.json")   # BEIR模式下的块文件
BEIR_QUESTION_FILE = os.path.join(_DATA_ROOT, "beir_题库.json")   # BEIR模式下的题库文件

QUESTION_FILE = "测试题库.json"     # PDF模式下的评测题库（含问题与"相关块编号"标注）
EXPECTED_CHUNK_COUNT = 14          # 对照表说明共14个块，编号0~13（仅PDF模式核对用）
QUERY_TOP_K = 50                    # 检索返回片段数，必须 ≥ TOP_K_LIST 的最大档位
QUERY_REPEAT = 3                   # 每道题重复检索次数（取平均延迟，首次预热不计入）
MAX_QUERIES = 0                    # 最多评测多少道题（0=全部；BEIR题库大时建议先取子集）
TOP_K_LIST = (1, 5, 10, 20, 50)            # 统计档位；BEIR每题相关块多，建议改为(1, 5, 10, 20, 50)并把QUERY_TOP_K调到50

# ======================== 以下为代码逻辑，无需改动 ========================

# 报告内容收集（控制台打印 + 写入UTF-8文件，避免乱码）
report_lines = []


def log(s=""):
    print(s, flush=True)   # 实时刷新：重定向到文件时也能及时看到进度
    report_lines.append(s)


def get_rss_mb():
    """当前评测进程的常驻内存（兆字节）；psutil未安装时返回None"""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def dir_size_mb(path):
    """目录总大小（兆字节）；目录不存在时返回0"""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def find_block_ids(retrieved_text, chunk_texts):
    """通过文本内容反查检索片段覆盖的块编号集合
    支持两种情况：
      - 片段就是某个块本身（稠密向量索引返回的就是块原文）
      - 片段包含多个块（父文档索引返回的是合并后的大段文本）
    """
    matched = set()
    for i, ct in enumerate(chunk_texts):
        if ct and (ct in retrieved_text or retrieved_text in ct):
            matched.add(i)
    return matched


def blocks_to_docs(block_ids, chunk_doc_ids):
    """块编号集合 → 来源文档id集合（仅BEIR模式；chunk_doc_ids为None时返回空集）"""
    if chunk_doc_ids is None:
        return set()
    return {chunk_doc_ids[i] for i in block_ids if 0 <= i < len(chunk_doc_ids)}


# --- 第一步：核对切分结果与对照表是否一致 ---
def check_chunks(chunk_texts):
    log("=" * 70)
    log("【核对阶段】与《块编号对照表.txt》比对")
    log("=" * 70)
    if len(chunk_texts) == EXPECTED_CHUNK_COUNT:
        log(f"块数核对通过：共 {len(chunk_texts)} 个块（编号0~{len(chunk_texts)-1}），与对照表一致")
    else:
        log(f"警告：实际 {len(chunk_texts)} 个块，对照表为 {EXPECTED_CHUNK_COUNT} 个，"
            f"可能PDF版本或切分参数有差异，评测编号可能对不上！")
    log("\n块0开头（应与对照表块0内容一致）：")
    log(chunk_texts[0][:120].replace("\n", " "))
    log("\n块11开头（对照表注明为噪声块，含版权信息）：")
    log(chunk_texts[11][:120].replace("\n", " "))


# --- 第二步：检索评测（含延迟测量） ---
def evaluate(questions, index_module, config, chunk_texts, chunk_doc_ids=None):
    log("=" * 70)
    log("【评测阶段】逐题检索并与题库标注比对")
    log("=" * 70)

    records = []           # 指标记录：dict(题id, 类别, Top-K, 召回率, 精确率, F1, 命中数)
    latency_records = []   # 延迟记录：tuple(题id, 平均毫秒, 最小毫秒, 最大毫秒)

    for q in questions:
        question = q["问题"]
        gold = q["相关块编号"]

        # 预热1次不计时：避免首次调用的初始化/缓存开销污染延迟测量
        index_module.search(question, config)

        # 重复检索取平均延迟（延迟含问题向量化耗时，即端到端查询延迟）
        times = []
        retrieved = []
        for _ in range(QUERY_REPEAT):
            t0 = time.perf_counter()
            result = index_module.search(question, config)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            retrieved = result
        latency_records.append(
            (q["id"], sum(times) / len(times) * 1000,
             min(times) * 1000, max(times) * 1000)
        )

        # 拒答题：标注块为空，不参与指标计算，仅展示检索情况
        if not gold:
            log(f"\n[{q['id']}] {q['问题']}")
            log(f"  类别：{q['类别']}（无标注相关块，仅展示检索结果，供人工判断）")
            for i, text in enumerate(retrieved, 1):
                ids = find_block_ids(text, chunk_texts)
                show = sorted(ids) if ids else "无"
                log(f"    第{i}名：匹配块{show}  开头：{text[:40].replace(chr(10), ' ')}...")
            continue

        gold_set = set(gold)
        # BEIR模式：块级标注映射为文档级相关文档集合（qrels本质是文档级标注，
        # 文档级命中判断才是BEIR官方的评测方式）
        gold_docs = blocks_to_docs(gold_set, chunk_doc_ids)

        gold_show = gold[:10] if len(gold) > 10 else gold
        log(f"\n[{q['id']}] {q['问题']}")
        log(f"  类别：{q['类别']}  标注块：{gold_show}{'...' if len(gold) > 10 else ''}")
        for i, text in enumerate(retrieved[:3], 1):
            ids = find_block_ids(text, chunk_texts)
            if chunk_doc_ids is not None:
                hit_docs = blocks_to_docs(ids, chunk_doc_ids) & gold_docs
                mark = "√命中" if hit_docs else ""
            else:
                mark = "√命中" if gold_set & ids else ""
            show = sorted(ids) if ids else "无"
            log(f"    第{i}名：匹配块{show}  {mark}")

        # 统计 TOP_K_LIST 中每个档位的指标
        for k in TOP_K_LIST:
            covered = set()
            for text in retrieved[:k]:
                covered |= find_block_ids(text, chunk_texts)
            if chunk_doc_ids is not None:
                # 文档级评测（BEIR标准）：
                #   召回率 = 命中的相关文档数 / 该题相关文档总数
                #   精确率 = 命中的相关片段数 / 检索返回片段数k
                covered_docs = blocks_to_docs(covered, chunk_doc_ids)
                hit = len(covered_docs & gold_docs)
                recall = hit / len(gold_docs)
                hit_frags = sum(
                    1 for text in retrieved[:k]
                    if blocks_to_docs(find_block_ids(text, chunk_texts),
                                      chunk_doc_ids) & gold_docs
                )
                precision = hit_frags / k
            else:
                hit = len(gold_set & covered)
                recall = hit / len(gold_set)
                precision = hit / k
            f1 = (2 * recall * precision / (recall + precision)
                  if (recall + precision) > 0 else 0.0)
            records.append({
                "id": q["id"], "类别": q["类别"], "k": k,
                "recall": recall, "precision": precision,
                "f1": f1, "hit": hit
            })

        covered_all = set()
        for text in retrieved:
            covered_all |= find_block_ids(text, chunk_texts)
        if chunk_doc_ids is not None:
            hit_all = len(blocks_to_docs(covered_all, chunk_doc_ids) & gold_docs)
            total = len(gold_docs)
        else:
            hit_all = len(gold_set & covered_all)
            total = len(gold_set)
        log(f"  召回率@{len(retrieved)} = {hit_all}/{total} = {hit_all / total:.2%}")
    return records, latency_records


# --- 第三步：汇总输出 ---
def summarize(records, title):
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)
    if not records:
        log("  （无数据）")
        return
    for k in TOP_K_LIST:
        sub = [r for r in records if r["k"] == k]
        n = len(sub)
        avg_recall = sum(r["recall"] for r in sub) / n
        avg_precision = sum(r["precision"] for r in sub) / n
        avg_f1 = sum(r["f1"] for r in sub) / n
        hit_rate = sum(1 for r in sub if r["hit"] > 0) / n
        log(f"  Top-{k}（共{n}题）：平均召回率={avg_recall:.2%}  "
            f"平均精确率={avg_precision:.2%}  F1={avg_f1:.2%}  "
            f"题目命中率={hit_rate:.2%}")


# --- 第三步（附加）：召回率集中摘要 ---
def summarize_recall(records, title):
    """把 Recall@5/10/20/50 集中在一行输出，跨方案对比时直接对照这一行即可"""
    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)
    if not records:
        log("  （无数据）")
        return
    parts = []
    for k in TOP_K_LIST:
        if k == 1:   # 摘要只输出 5/10/20/50 四个档位（@1 命中率低、波动大，不作对比重点）
            continue
        sub = [r for r in records if r["k"] == k]
        avg = sum(r["recall"] for r in sub) / len(sub)
        parts.append(f"Recall@{k}={avg:.2%}")
    log("  " + "   ".join(parts))


def summarize_latency(latency_records):
    log("")
    log("=" * 70)
    log(f"【查询延迟统计】（每道题预热1次后重复测{QUERY_REPEAT}次取平均，单位毫秒）")
    log("=" * 70)
    for qid, avg_ms, min_ms, max_ms in latency_records:
        log(f"  [{qid}] 平均 {avg_ms:.1f} 毫秒（最小 {min_ms:.1f} / 最大 {max_ms:.1f}）")
    all_avg = [r[1] for r in latency_records]
    if all_avg:
        log(f"  总体：平均 {sum(all_avg) / len(all_avg):.1f} 毫秒，"
            f"最慢单题 {max(all_avg):.1f} 毫秒")
    log("  说明：延迟包含问题向量化的耗时，即端到端查询延迟；"
        "不含Ollama服务的进程启动时间（首次已预热）")


def summarize_resources(build_time, rss_before, rss_after, disk_before, disk_after):
    log("")
    log("=" * 70)
    log("【资源占用】")
    log("=" * 70)
    log(f"  建库耗时：{build_time:.2f} 秒")
    if rss_before is None:
        log("  内存占用：未安装 psutil，已跳过测量（安装命令：pip install psutil）")
    else:
        log(f"  内存占用：建库前 {rss_before:.1f} MB → 建库后 {rss_after:.1f} MB，")
        log(f"            本次建库内存增量约 {rss_after - rss_before:.1f} MB")
    log(f"  磁盘占用：存储目录 建库前 {disk_before:.2f} MB → 建库后 {disk_after:.2f} MB，")
    log(f"            本方案本次写入约 {disk_after - disk_before:.2f} MB")
    log("  说明：内存仅统计评测进程（不含Ollama嵌入服务进程）；磁盘增量为本次建库")
    log("        写入量，多方案并存时目录总量会累加。")


def main():
    skip_build = "--skip-build" in sys.argv

    # 校验配置：检索返回数必须覆盖最大统计档位
    assert QUERY_TOP_K >= max(TOP_K_LIST), \
        f"配置错误：QUERY_TOP_K({QUERY_TOP_K}) 必须 ≥ TOP_K_LIST 的最大档位({max(TOP_K_LIST)})"

    # 1. 加载与主程序一致的索引方案（主程序里改 INDEX_METHOD 即切换评测对象）
    index_module = rag_demo.load_index_module()
    config = dict(rag_demo.INDEX_CONFIG)   # 复制主程序公共配置，避免互相影响
    config["n_results"] = QUERY_TOP_K      # 评测统一取前QUERY_TOP_K个片段

    report_file = f"评测报告_{rag_demo.INDEX_METHOD}_{DATA_MODE}.txt"  # 每个方案/数据源一份报告

    log("=" * 70)
    log(f"检索评测报告 —— 索引方案：{rag_demo.INDEX_METHOD}   数据源：{DATA_MODE}    "
        f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Top-K：{QUERY_TOP_K}   统计档位：{list(TOP_K_LIST)}")
    log("=" * 70)

    # 2. 数据准备：按数据源模式加载块文本
    if DATA_MODE == "beir":
        # BEIR模式：加载 prepare_beir.py 生成的块文件（含来源文档id，用于文档级评测）
        with open(BEIR_CHUNKS_FILE, encoding="utf-8") as f:
            chunk_items = json.load(f)
        chunk_texts = [item["text"] for item in chunk_items]
        chunk_doc_ids = [item.get("doc_id") for item in chunk_items]
        # 包装成Document对象，适配索引方案接口 build_index(chunks, config)
        chunks = [Document(page_content=t, metadata={"chunk": i})
                  for i, t in enumerate(chunk_texts)]
        log(f"BEIR模式：已加载 {len(chunk_texts)} 个块（来自 {BEIR_CHUNKS_FILE}）")
    else:
        # PDF模式：加载主程序配置的PDF并切分
        chunks = rag_demo.load_and_split_pdf(rag_demo.PDF_FILE)
        chunk_texts = [c.page_content for c in chunks]
        chunk_doc_ids = None
        check_chunks(chunk_texts)

    # 3. 建库并测量资源占用（默认重建保证评测可复现；--skip-build 跳过）
    if skip_build:
        log("\n已指定 --skip-build，跳过建库，直接评测现有索引")
    else:
        log("")
        log("【建库阶段】")
        rss_before = get_rss_mb()
        disk_before = dir_size_mb(rag_demo.STORAGE_PATH)
        t0 = time.perf_counter()
        index_module.build_index(chunks, config)
        build_time = time.perf_counter() - t0
        rss_after = get_rss_mb()
        disk_after = dir_size_mb(rag_demo.STORAGE_PATH)
        summarize_resources(build_time, rss_before, rss_after, disk_before, disk_after)

    # 4. 加载题库并逐题评测
    question_file = BEIR_QUESTION_FILE if DATA_MODE == "beir" else QUESTION_FILE
    with open(question_file, encoding="utf-8") as f:
        questions = json.load(f)
    total_q = len(questions)
    if MAX_QUERIES > 0 and total_q > MAX_QUERIES:
        questions = questions[:MAX_QUERIES]
        log(f"\n已按 MAX_QUERIES 截取前 {MAX_QUERIES} 道题评测（题库共 {total_q} 道）")

    records, latency_records = evaluate(questions, index_module, config, chunk_texts, chunk_doc_ids)

    summarize_latency(latency_records)

    # 汇总：全部有标注的题
    summarize(records, f"全部有标注题汇总（{len(records) // len(TOP_K_LIST)}题，不含拒答题）")

    # 分类别汇总（PDF模式按题库类别，BEIR模式按数据集名）
    for cat in dict.fromkeys(r["类别"] for r in records):
        cat_records = [r for r in records if r["类别"] == cat]
        if cat_records:
            summarize(cat_records, f"类别【{cat}】汇总（{len(cat_records) // len(TOP_K_LIST)}题）")

    # 召回率集中摘要：各档位 Recall 一行输出（跨方案对比时直接对照这一行即可）
    summarize_recall(records, "召回率摘要（跨方案对比，全部有标注题）")

    # 拒答题检索情况说明
    log("")
    log("=" * 70)
    log("拒答题说明")
    log("=" * 70)
    log("拒答题在数据集中本就没有相关块，不参与召回率/精确率统计。")
    log("判断标准：若拒答题检索到的片段与标注块完全无关（匹配块编号与正常题重叠少），")
    log("说明系统没有强行召回不相关内容，拒答表现良好；反之可能误导大模型编造答案。")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\n评测报告已保存到 {report_file}")


if __name__ == "__main__":
    main()
