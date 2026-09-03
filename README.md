# RAG 索引评测框架（RAG Index Benchmark）

一个可插拔索引方案的检索增强生成（RAG）系统与检索评测框架：以 BEIR 基准为评测数据源，对不同的索引方案（稠密向量、稀疏检索、混合检索、父文档索引……）进行统一的召回率、精确率、延迟与资源占用评测，并支持一键切换索引方案运行问答。

- **本地嵌入**：Ollama + bge-m3（多语言嵌入模型），无需云端嵌入 API
- **本地索引**：FAISS 向量索引，向量数据落盘本地
- **生成模型**：DeepSeek（仅问答模式需要）
- **评测基准**：BEIR（nfcorpus），自动将文档级标注转换为块级题库

## 核心特点

1. **可插拔索引方案**：每个索引方案是 [indexes/](indexes/) 目录下的一个独立脚本，只需实现 `build_index(chunks, config)` 和 `search(question, config)` 两个函数；主程序通过配置区 `INDEX_METHOD` 一键切换，主程序零改动（接口约定见 [indexes/README.md](indexes/README.md)）。
2. **完整的评测闭环**：逐题检索并自动比对标注，输出多档位（Top-1/5/10/20/50）召回率、精确率、F1、题目命中率，以及建库耗时、查询延迟、内存/磁盘占用。
3. **评测与运行同配置**：评测脚本直接复用主程序的配置与索引方案加载逻辑，不存在"评测一套、运行一套"的配置不一致问题。
4. **两种数据源模式**：BEIR 模式（`prepare_beir.py` 自动下载数据集并生成块级题库）与 PDF 模式（本地 PDF + 手工标注题库）。

## 架构图

```mermaid
graph TB
    subgraph 入口脚本
        RAG["rag_demo.py<br/>公共配置中心 + 问答主流程"]
        PREP["prepare_beir.py<br/>BEIR 数据准备（下载/切块/生成题库）"]
        EVAL["eval_retrieval.py<br/>检索评测（召回率/F1/延迟/资源）"]
    end

    subgraph 索引方案插件["indexes/（可插拔，文件名即方案名）"]
        IDX["dense.py<br/>稠密向量索引（FAISS）✔"]
        BM25["bm25.py<br/>BM25 稀疏检索（规划中）"]
        HYB["hybrid.py<br/>混合检索 RRF（规划中）"]
        PAR["parent_doc.py<br/>父文档索引（规划中）"]
    end

    subgraph 外部服务
        OLLAMA["Ollama 本地服务<br/>bge-m3 嵌入模型"]
        DS["DeepSeek API<br/>回答生成（可选）"]
    end

    subgraph 数据与存储
        CHUNKS["beir_chunks.json<br/>全部块文本"]
        BANK["beir_题库.json<br/>题目 + 相关块标注"]
        STORE["vector_store/&lt;方案名&gt;/<br/>索引文件 + 文本"]
    end

    PREP -->|"复用 rag_demo 切分参数"| RAG
    PREP --> CHUNKS
    PREP --> BANK
    RAG -->|"动态加载 INDEX_METHOD"| IDX
    EVAL -->|"复用 rag_demo 配置与加载逻辑"| RAG
    EVAL -->|"动态加载 INDEX_METHOD"| IDX
    IDX -->|"向量化"| OLLAMA
    RAG -->|"生成回答"| DS
    IDX --> STORE
    CHUNKS --> RAG
    CHUNKS --> EVAL
    BANK --> EVAL
```

**一次问答的数据流：**

```mermaid
flowchart LR
    Q[用户问题] --> S["索引方案 search()"]
    S --> V["问题向量化<br/>（Ollama bge-m3）"]
    V --> F["FAISS 相似度检索<br/>（余弦相似度）"]
    F --> C["Top-K 相关片段"]
    C --> L["拼接提示词<br/>DeepSeek 生成"]
    L --> A[回答]
```

## 快速开始

### 环境要求

| 依赖 | 说明 |
|---|---|
| Python 3.11+ | 运行环境 |
| Ollama | 本地嵌入服务，需 `ollama pull bge-m3` |
| DeepSeek API Key | 可选，仅"生成回答"模式需要；纯检索评测不需要 |

### 安装与运行

```bash
# 1. 安装 Python 依赖
pip install openai langchain-community langchain-text-splitters langchain-core faiss-cpu numpy certifi
# 可选：评测内存占用需要 psutil
pip install psutil

# 2. 启动 Ollama 并拉取嵌入模型（需先安装 Ollama 并启动服务）
ollama pull bge-m3

# 3. 准备 BEIR 数据（自动下载 nfcorpus 数据集，切块并生成块级题库）
python prepare_beir.py

# 4. 构建索引并跑完整评测（输出 评测报告_dense_beir.txt）
python eval_retrieval.py

# 5. 运行问答示例（需先设置 DEEPSEEK_API_KEY 环境变量）
python rag_demo.py
```

### 常用参数（在 [rag_demo.py](rag_demo.py) 配置区修改）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `INDEX_METHOD` | `"dense"` | 索引方案名 = `indexes/` 下的文件名（不含 .py） |
| `RETRIEVE_ONLY` | `False` | 置 `True` 只打印检索片段不调大模型，快速对比方案召回 |
| `N_RESULTS` | `3` | 问答模式召回片段数 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | 文档切分参数 |
| `EMBEDDING_MODEL` | `"bge-m3"` | Ollama 中的嵌入模型名 |

评测参数（召回档位、查询重复次数、数据源模式等）在 [eval_retrieval.py](eval_retrieval.py) 配置区修改。

## 评测结果（基线：dense 方案）

**评测环境**：nfcorpus 数据集 ｜ 语料 3633 篇文档切为 17071 块 ｜ 323 道题 ｜ 嵌入模型 bge-m3（本地 CPU）｜ 检索方式 FAISS 暴力检索（IndexFlatIP + L2 归一化，即余弦相似度）

**评测口径**：BEIR 的 qrels 为文档级标注，本框架按官方方式做**文档级命中判断**——检索片段经"文本反查块编号 → 块编号映射来源文档"后与相关文档比对。nfcorpus 平均每题约 38 篇相关文档（块级相关块被截断至最多 50 个），文档级召回率天然低于块级口径。

| Top-K | 平均召回率 | 平均精确率 | F1 | 题目命中率 |
|------:|----------:|----------:|----:|----------:|
| 1  | 5.31% | 35.91% | 7.78% | 35.91% |
| 5  | 11.11% | 27.31% | 12.51% | 54.80% |
| 10 | 13.63% | 22.45% | 13.31% | 62.23% |
| 20 | 16.78% | 17.55% | 13.60% | 70.59% |
| 50 | 20.70% | 11.81% | 12.30% | 77.40% |

**资源与延迟**（2026-08-29 实测，完整报告见 [评测报告_dense_beir.txt](评测报告_dense_beir.txt)）：

| 指标 | 数值 | 说明 |
|---|---|---|
| 建库耗时 | 220.27 秒 | 含 17071 块本地 CPU 向量化 |
| 内存增量 | 78.0 MB | 评测进程建库前后差（不含 Ollama 进程） |
| 磁盘占用 | 约 72.7 MB | `vector_store/faiss_dense/`（index.faiss + texts.json） |
| 平均查询延迟 | 1948.8 毫秒 | 端到端（含问题向量化），预热后 3 次取平均 |

> 说明：平均延迟主要来自 bge-m3 在本地 CPU 上的问题向量化；FAISS 暴力检索本身为 O(N·d) 全量扫描，随着语料规模增大，这正是后续引入 IVF/HNSW 近似检索与量化压缩的优化空间（见 Roadmap）。

## 添加新的索引方案

1. 复制 [indexes/dense.py](indexes/dense.py)，改名为你的方案名（如 `parent_doc.py`）；
2. 实现 `build_index(chunks, config)` 与 `search(question, config)` 两个函数（接口约定见 [indexes/README.md](indexes/README.md)）；
3. 在 [rag_demo.py](rag_demo.py) 配置区把 `INDEX_METHOD` 改为你的文件名；
4. 运行 `python eval_retrieval.py` 得到该方案的独立评测报告，与已有方案对比。

## 项目结构

```
RAG/
├── rag_demo.py              # 主程序：公共配置中心 + 问答流程
├── eval_retrieval.py        # 检索评测脚本（多指标 + 报告输出）
├── prepare_beir.py          # BEIR 数据准备（下载/切块/块级题库生成）
├── indexes/                 # 可插拔索引方案目录
│   ├── README.md            # 方案接口约定
│   └── dense.py             # 稠密向量索引方案（FAISS）
├── vector_store/            # 向量索引存储根目录（各方案独立子目录）
├── beir_data/               # BEIR 数据集缓存
├── beir_chunks.json         # 块文件（prepare_beir.py 生成）
├── beir_题库.json            # 块级标注题库（prepare_beir.py 生成）
└── 评测报告_dense_beir.txt   # 基线评测报告
```

## Roadmap

- [x] 稠密向量索引（FAISS 暴力检索 + 余弦相似度）
- [x] BEIR 基准评测闭环（召回率/精确率/F1/延迟/资源占用）
- [ ] BM25 稀疏检索方案（关键词倒排）
- [ ] 混合检索方案（BM25 + 稠密向量，RRF 分数融合）
- [ ] 父文档索引方案
- [ ] 近似最近邻检索（IVF / HNSW）与召回率-延迟曲线
- [ ] 向量量化压缩（PQ / 标量量化）与内存-磁盘占用对比

## 许可证

[MIT](LICENSE)
