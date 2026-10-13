# -*- coding: utf-8 -*-
"""trec-covid 60 万块向量化（带断点续跑）

背景：eval_retrieval.py 的 dense 建库向量化无断点续跑，Ollama 中途 500 崩溃时
进度全丢（2026-10-13 曾跑到 42% 后丢失）。本脚本用 memmap 增量写盘：每批向量化
后立即写入对应切片并记录进度，中断重跑自动从断点继续。完成后构建 IndexFlatIP
并落盘到 vector_store/faiss_dense/（与 indexes/dense.py 完全一致的形态）。

用法：
  python experiments/build_trec_index.py
中断后重跑同一命令即可续跑。
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss
import numpy as np

from openai import OpenAI   # noqa: E402  失败时重建连接（见 main 内说明）


def fresh_client():
    """新建嵌入客户端（新连接池）。Ollama 的 tokenizer 子服务崩溃后，keep-alive
    连接仍指向死服务、后续请求全 400——失败必须销毁客户端重建连接（2026-10-13 定位）"""
    return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

# 脚本在 experiments/ 下：上一级=项目目录，再上一级=数据根目录（D:/file/RAG）
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS_FILE = os.path.join(DATA_ROOT, "beir_chunks_trec-covid.json")
WORK_DIR = os.path.join(DATA_ROOT, "vector_store", "_trec_build")   # 中间产物目录
VECS_PATH = os.path.join(WORK_DIR, "vecs.bin")
DONE_PATH = os.path.join(WORK_DIR, "done.txt")
OUT_DIR = os.path.join(DATA_ROOT, "vector_store", "faiss_dense")

BATCH = 256
DIM = 1024
MAX_RETRY = 5          # 每批重试次数（500 Compute error 偶发，退避重试）


def main() -> None:
    print("trec-covid 向量化（断点续跑）")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        items = json.load(f)
    texts = [it["text"] for it in items]
    n = len(texts)
    print(f"块数：{n}")

    os.makedirs(WORK_DIR, exist_ok=True)
    mm = np.memmap(VECS_PATH, dtype=np.float32, mode="w+", shape=(n, DIM))

    done = 0
    if os.path.exists(DONE_PATH):
        done = int(open(DONE_PATH).read().strip())
        print(f"检测到断点：已完成 {done}/{n}，继续...")

    t_start = time.perf_counter()
    client = fresh_client()
    for i in range(done, n, BATCH):
        end = min(i + BATCH, n)
        batch = texts[i:end]
        resp = None
        for attempt in range(MAX_RETRY):
            try:
                resp = client.embeddings.create(model="bge-m3", input=batch)
                break
            except Exception as e:
                if attempt == MAX_RETRY - 1:
                    raise
                # 关键：销毁连接池重建客户端（keep-alive 死连接必须换新，仅等待无效）
                try:
                    client.close()
                except Exception:
                    pass
                client = fresh_client()
                wait = 5 * (attempt + 1)
                print(f"  第{i}~{end}批失败（{e}），重建连接后 {wait} 秒重试...", flush=True)
                time.sleep(wait)
        vecs = np.array([it.embedding for it in resp.data], dtype=np.float32)
        mm[i:end] = vecs
        mm.flush()
        done = end
        with open(DONE_PATH, "w") as fp:
            fp.write(str(done))
        if (end // BATCH) % 20 == 0:
            el = time.perf_counter() - t_start
            rate = (end - (done - (end - i))) / el   # 本段速率粗估
            print(f"  进度：{end}/{n}（{el:.0f} 秒累计）", flush=True)

    mm.flush()
    print("向量化完成，构建索引...")
    matrix = np.asarray(mm)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(DIM)
    index.add(matrix)
    os.makedirs(OUT_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    with open(os.path.join(OUT_DIR, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print(f"索引已落盘 {OUT_DIR}（{n} 块）")


if __name__ == "__main__":
    main()
