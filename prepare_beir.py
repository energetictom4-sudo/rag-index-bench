# -*- coding: utf-8 -*-
"""
BEIR 数据集准备脚本：下载并转换为评测脚本可用的块级数据

功能：
  1. 下载 BEIR 数据集（zip 包，托管在 UKP 官方服务器）
  2. 读取语料库(corpus)、查询(queries)、相关标注(qrels)
  3. 用与主程序完全一致的切分参数把语料文档切成块，记录"文档→块编号"映射
  4. 把文档级标注(qrels)转换为块级标注，生成评测题库
  5. 输出两个文件，供 eval_retrieval.py 使用：
     - beir_chunks.json    所有块的文本
     - beir_题库.json      块级标注题库

用法：
  python prepare_beir.py                 # 按配置区 DATASET 下载并转换
  python prepare_beir.py --skip-download # 数据集已下载过则跳过下载
  python prepare_beir.py --force         # 强制重新下载（覆盖旧数据）

数据集选择建议（按语料规模从小到大，本地CPU跑bge-m3向量化耗时与规模成正比）：
  nfcorpus     3.6k文档  |  323查询   最小，首选
  scifact      5.2k文档  |  300查询   科学事实核查
  arguana      8.7k文档  |  1406查询  论证检索
  scidocs      25.7k文档 |  1000查询  科学文献
  fiqa         57.6k文档 |  648查询   金融问答
  trec-covid   171k文档  |  50查询    生物医学
  msmarco      8.8M文档  |  本地不推荐（向量化太慢，先用小数据集）

说明：
  - BEIR 的 qrels 是文档级标注，本脚本自动转换为块级：某题标注了文档A，
    则文档A切出的所有块都算该题的相关块
  - BEIR 以英文数据为主（嵌入模型 bge-m3 支持多语言，英文可用）；
    若坚持中文数据可考虑 MIRACL 基准，不在本脚本范围内
  - 若官方服务器下载失败，可手动下载
    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip
    放入 beir_data/ 目录后加 --skip-download 运行
"""

import json
import os
import random
import sys
import urllib.request
import zipfile

# 解决 Windows 控制台中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

# 复用主程序的切分参数（保持与主程序/评测完全一致）
import rag_demo
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================ 参数配置区 ================

DATASET = "nfcorpus"        # 数据集名（见上方建议清单）
BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATA_DIR = "./beir_data"    # 数据集存放目录（zip与解压后的数据）
OUT_CHUNKS = "beir_chunks.json"     # 输出的块文件
OUT_QUESTIONS = "beir_题库.json"    # 输出的块级题库
MAX_DOCS = 0                # 最多加载多少篇语料文档（0=全部；语料太大时可先取子集快速实验，
                            #   注意：被截断的文档对应的题会变成无相关块的"拒答题"）
MAX_QUERIES = 0             # 最多输出多少道题（0=全部；查询太多时可先取子集）
MAX_REL_BLOCKS = 50         # 每道题最多保留的相关块数：BEIR每题相关文档多（如nfcorpus平均38篇），
                            # 文档级标注转块级后相关块可达数百个，会严重稀释召回率；
                            # 超过该数量时用固定随机种子采样截断，让Top-50档位的召回率有意义（0=不截断）

# ======================== 以下为代码逻辑，无需改动 ========================

# 解决Windows常见的"certificate verify failed"错误：
# 系统证书链不完整时，改用certifi打包的Mozilla CA证书库（证书验证仍然开启，只是换用完整证书链）
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass   # certifi未安装则保持系统默认行为


def download_dataset(force: bool = False) -> str:
    """下载并解压 BEIR 数据集，返回解压后的目录"""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"{DATASET}.zip")
    extract_dir = os.path.join(DATA_DIR, DATASET)

    if force or not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
        url = f"{BEIR_BASE_URL}/{DATASET}.zip"
        print(f"正在下载 {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成，正在解压...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        print("解压完成")
    else:
        print(f"数据集 {DATASET} 已存在，跳过下载")
    return extract_dir


def load_jsonl(path: str) -> list[dict]:
    """按行读取 jsonl 文件，返回列表"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    force = "--force" in sys.argv
    skip_download = "--skip-download" in sys.argv

    # 1. 下载并解压数据集
    if skip_download:
        extract_dir = os.path.join(DATA_DIR, DATASET)
        if not os.path.isdir(extract_dir):
            raise SystemExit(f"数据集目录 {extract_dir} 不存在，请先去掉 --skip-download 运行")
    else:
        extract_dir = download_dataset(force=force)

    # 2. 读取语料库
    corpus = load_jsonl(os.path.join(extract_dir, "corpus.jsonl"))
    if MAX_DOCS > 0 and len(corpus) > MAX_DOCS:
        corpus = corpus[:MAX_DOCS]
        print(f"已按 MAX_DOCS 截取前 {MAX_DOCS} 篇文档（被截断文档对应的题将无相关块）")
    print(f"语料共 {len(corpus)} 篇文档")

    # 3. 用与主程序一致的切分器把每篇文档切成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_demo.CHUNK_SIZE,
        chunk_overlap=rag_demo.CHUNK_OVERLAP,
        separators=rag_demo.SEPARATORS
    )
    chunk_texts = []        # 块文本列表（下标即块编号，与评测脚本的匹配逻辑一致）
    chunk_doc_ids = []      # 每块的来源文档id（供评测做文档级命中判断）
    doc_to_chunks = {}      # 文档id → 该文档切出的块编号列表
    for doc in corpus:
        doc_id = doc["_id"]
        text = (doc.get("text") or "").strip()
        title = (doc.get("title") or "").strip()
        full = f"{title}\n{text}" if title else text
        if not full:
            continue
        pieces = text_splitter.split_text(full)
        start = len(chunk_texts)
        chunk_texts.extend(pieces)
        chunk_doc_ids.extend([doc_id] * len(pieces))
        doc_to_chunks[doc_id] = list(range(start, start + len(pieces)))
    avg_blocks = len(chunk_texts) / len(doc_to_chunks) if doc_to_chunks else 0
    print(f"共切出 {len(chunk_texts)} 个块（每篇文档平均 {avg_blocks:.1f} 块）")

    # 4. 读取查询与文档级标注(qrels)，转换为块级标注
    queries = load_jsonl(os.path.join(extract_dir, "queries.jsonl"))
    query_map = {q["_id"]: q["text"] for q in queries}
    qrels = {}   # query_id → 相关文档id列表
    # 兼容两种qrels格式（不同BEIR数据集可能不同）：
    #   三列格式：query-id, corpus-id, score（如nfcorpus）
    #   四列TREC格式：query-id, Q0, corpus-id, score（如msmarco）
    with open(os.path.join(extract_dir, "qrels", "test.tsv"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts or parts[0] in ("query-id", "query_id", "qid"):
                continue   # 跳过表头行
            if len(parts) == 4:
                qid, doc_id, score = parts[0], parts[2], parts[3]
            elif len(parts) == 3:
                qid, doc_id, score = parts[0], parts[1], parts[2]
            else:
                continue
            if int(score) == 2:   # BEIR官方口径：nfcorpus只把强相关(score=2)算相关
                qrels.setdefault(qid, []).append(doc_id)

    questions = []
    qid_list = list(qrels.keys())
    if MAX_QUERIES > 0 and len(qid_list) > MAX_QUERIES:
        qid_list = qid_list[:MAX_QUERIES]
        print(f"已按 MAX_QUERIES 截取前 {MAX_QUERIES} 道题")
    rng = random.Random(42)   # 固定种子：保证每次生成的截断结果一致，评测可复现
    for qid in qid_list:
        block_ids = set()
        for doc_id in qrels[qid]:
            block_ids.update(doc_to_chunks.get(doc_id, []))
        block_ids = sorted(block_ids)
        # 相关块过多时按固定种子随机采样截断，避免召回率被稀释
        if 0 < MAX_REL_BLOCKS < len(block_ids):
            block_ids = sorted(rng.sample(block_ids, MAX_REL_BLOCKS))
        questions.append({
            "id": qid,
            "问题": query_map.get(qid, qid),
            "类别": DATASET,            # 分类汇总时按数据集名归为一类
            "相关块编号": block_ids
        })

    # 5. 保存输出文件
    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump([{"id": i, "text": t, "doc_id": d}
                   for i, (t, d) in enumerate(zip(chunk_texts, chunk_doc_ids))],
                  f, ensure_ascii=False, indent=1)
    with open(OUT_QUESTIONS, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=1)

    # 6. 统计信息
    no_rel = sum(1 for q in questions if not q["相关块编号"])
    rel_counts = [len(q["相关块编号"]) for q in questions if q["相关块编号"]]
    avg_rel = sum(rel_counts) / len(rel_counts) if rel_counts else 0
    print(f"\n完成！共生成 {len(questions)} 道题（其中 {no_rel} 道无相关块，作为拒答题参考）")
    print(f"每道题平均相关块数：{avg_rel:.1f}（设置 eval_retrieval.py 的 TOP_K_LIST 档位时参考）")
    if MAX_REL_BLOCKS > 0:
        print(f"已按 MAX_REL_BLOCKS 截断：每道题最多保留 {MAX_REL_BLOCKS} 个相关块（固定种子随机采样，可复现）")
    print(f"块文件已保存到 {OUT_CHUNKS}")
    print(f"题库已保存到 {OUT_QUESTIONS}")
    print("\n下一步：")
    print("  1. eval_retrieval.py 配置区按上面平均相关块数调整 QUERY_TOP_K / TOP_K_LIST")
    print("  2. rag_demo.py 配置区设置要评测的 INDEX_METHOD")
    print("  3. 运行 python eval_retrieval.py")


if __name__ == "__main__":
    main()
