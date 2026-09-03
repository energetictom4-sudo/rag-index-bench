# 索引方案接口约定

主程序 [rag_demo.py](../rag_demo.py) 通过配置区参数 `INDEX_METHOD` 决定加载哪个索引方案。
每个索引方案 = 本目录下的一个 `.py` 文件，**文件名（不含.py后缀）即方案名**。

## 必须实现的两个函数

每个索引方案脚本必须实现以下两个函数（函数名必须完全一致）：

### 1. build_index(chunks, config)

构建索引。

- `chunks`：主程序切分好的文本片段列表，每个元素是 langchain 的 Document 对象
  - 取文本用 `chunk.page_content`
  - 取元数据用 `chunk.metadata`（如来源页码）
- `config`：主程序传入的公共资源字典，字段如下：

| 字段 | 含义 |
|---|---|
| `config["ollama_client"]` | 嵌入客户端（OpenAI兼容格式），调用 `.embeddings.create(...)` 做向量化 |
| `config["embedding_model"]` | 嵌入模型名 |
| `config["collection_name"]` | 集合名，**建议加后缀**（如 `+ "_parent_doc"`）避免不同方案数据互相覆盖 |
| `config["n_results"]` | 期望召回的片段数量 |
| `config["storage_path"]` | 向量数据存储根目录，各方案在下面建自己的子目录 |

### 2. search(question, config)

检索。

- `question`：用户问题的原始文本（向量化等处理由方案自己决定）
- `config`：同上
- **返回**：相关的文本片段列表（`list[str]`），数量建议不超过 `config["n_results"]`

## 添加新索引方案的步骤

1. 复制 [dense.py](dense.py)，改名为你的方案名（如 `parent_doc.py`）
2. 按上面接口改写 `build_index` / `search` 的内部逻辑
3. 主程序配置区把 `INDEX_METHOD` 改成你的文件名
4. 运行对比效果（把 `RETRIEVE_ONLY` 设为 `True` 可只看检索结果、不调用大模型，对比更快更省钱）

## 注意事项

- 构建索引时建议先删除旧集合再重建，保证每次跑都拿到最新数据（参考 dense.py 写法）
- 函数名写错或缺失，主程序启动时会给出明确报错提示
- 不一定要用 Chroma：任何能实现"构建索引 + 检索"的方案都可以（如关键词倒排索引、父文档索引、知识图谱等），config 里用不到的字段忽略即可
