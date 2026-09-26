import os
import sys
import json
import importlib
from types import ModuleType
from openai import OpenAI, OpenAIError
from langchain_core.documents import Document   # 包装块文本用

# ================ 参数配置区（以后只需要改这里） ================

# --- 索引方案参数 ---
INDEX_METHOD = "parent_doc"  # 索引方案名称 = indexes/ 目录下的脚本文件名（不含.py后缀）
                          # 想测试对比哪个方案，就把这里改成对应文件名，例如：
                          #   "dense"       稠密向量索引（语义检索，见 indexes/dense.py）
                          #   "bm25"        稀疏关键词检索（倒排索引，见 indexes/bm25.py）
                          #   "doc_dense"   文档级索引+先搜文档再取块（见 indexes/doc_dense.py）
                          #   "hybrid"      BM25+dense 的 RRF 混合检索（见 indexes/hybrid.py）
                          #   "parent_doc"  父文档索引（小块检索大块返回，见 indexes/parent_doc.py）
                          # 接口约定见 indexes/README.md，写新方案不用改主程序
RETRIEVE_ONLY = False     # True = 只打印检索到的片段（快速对比不同索引方案的召回效果），不调用大模型

# --- 对话大模型参数（DeepSeek） ---
DEEPSEEK_API_KEY = None       # DeepSeek密钥；留空(None)则自动读取环境变量 DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = "https://api.deepseek.com"   # DeepSeek官方接口地址
LLM_MODEL = "deepseek-chat"   # 生成回答的大模型名称
TEMPERATURE = 0.1             # 生成随机性，越低回答越稳定

# --- 嵌入模型参数（本地Ollama，OpenAI兼容格式） ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"   # Ollama本地接口地址
OLLAMA_API_KEY = "ollama"     # Ollama本地服务不需要真实密钥，随便填一个
EMBEDDING_MODEL = "bge-m3"    # Ollama中已拉取的嵌入模型名（中文效果较好）

# --- 数据根目录（路径配置，先于下面的参数定义）---
# 数据（beir_chunks.json、beir_题库.json、vector_store、beir_data）都在项目目录的
# 上一级，与 experiments/common.py 的 DATA_ROOT 一致。用基于脚本位置的绝对路径，
# 保证无论从哪个目录运行主程序/评测脚本，路径都不会漂移。
_DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 向量存储参数 ---
STORAGE_PATH = os.path.join(_DATA_ROOT, "vector_store")   # 向量数据存储根目录（每个索引方案在里面建自己的子目录）
COLLECTION_NAME = "my_knowledge_base"   # 集合名称（每个索引方案会自动加后缀区分，见 indexes/dense.py）

# --- 文档切分参数 ---
CHUNK_SIZE = 500              # 每块最大500字符
CHUNK_OVERLAP = 50            # 块与块之间重叠50字符，保持上下文连贯
SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]

# --- 检索参数 ---
N_RESULTS = 3                 # 召回片段数量(Top-3)

# --- 数据源参数 ---
BEIR_CHUNKS_FILE = os.path.join(_DATA_ROOT, "beir_chunks.json")   # BEIR块文件（prepare_beir.py 生成）

# --- 运行示例参数 ---
USER_QUESTION = "Do Cholesterol Statin Drugs Cause Breast Cancer?"   # 要问的问题（可填 beir_题库.json 里的任意题）

# ======================== 以下为代码逻辑（一般不用改） ========================

# 1. 初始化公共资源（索引方案脚本通过 INDEX_CONFIG 共享使用）
# DeepSeek负责生成回答，Ollama负责文本向量化；向量存储由各索引方案脚本自行管理
# 注意：DeepSeek密钥缺失时不在这里报错，而是延迟到真正生成回答时才报，
#      这样评测/数据准备脚本（不需要对话大模型）也能正常导入本模块复用配置
try:
    deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
except OpenAIError:
    deepseek_client = None
ollama_client = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)

# 打包成配置字典传给索引方案脚本（字段说明见 indexes/README.md）
INDEX_CONFIG = {
    "ollama_client": ollama_client,      # 嵌入客户端（OpenAI兼容格式），用于文本向量化
    "embedding_model": EMBEDDING_MODEL,  # 嵌入模型名
    "collection_name": COLLECTION_NAME,  # 集合名，建议脚本加后缀避免不同方案数据互相覆盖
    "n_results": N_RESULTS,              # 期望召回的片段数量
    "storage_path": STORAGE_PATH,        # 向量数据存储根目录（各方案在下面建自己的子目录）
}


def load_index_module() -> ModuleType:
    """动态加载索引方案脚本（由 INDEX_METHOD 决定加载哪一个）"""
    # 把本脚本所在目录加入模块搜索路径，保证无论从哪里运行都能找到 indexes 包
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        module = importlib.import_module(f"indexes.{INDEX_METHOD}")
    except ModuleNotFoundError as e:
        # 区分两种失败：脚本文件本身不存在 / 脚本存在但它的依赖模块导入失败
        # （后者最常见的坑是脚本依赖的公共模块不在搜索路径中，误报“不存在”会误导排查方向）
        script_file = os.path.join(script_dir, "indexes", f"{INDEX_METHOD}.py")
        if not os.path.exists(script_file):
            raise SystemExit(
                f"未找到索引方案脚本 indexes/{INDEX_METHOD}.py，"
                f"请先在 indexes/ 目录创建该文件（接口约定见 indexes/README.md，示例见 indexes/dense.py）"
            )
        raise SystemExit(
            f"索引方案脚本 indexes/{INDEX_METHOD}.py 已找到，"
            f"但导入其依赖模块时失败：{e}\n"
            f"请检查该脚本的 import 语句：它依赖的公共模块是否存在于可被搜索到的路径"
        )

    # 校验脚本是否实现了约定的两个函数，避免函数名写错导致后续报错看不懂
    for func_name in ("build_index", "search"):
        if not hasattr(module, func_name):
            raise SystemExit(
                f"索引方案 indexes/{INDEX_METHOD}.py 缺少必须实现的函数 {func_name}()，"
                f"请参考 indexes/dense.py 的写法"
            )
    return module


def load_beir_chunks() -> list[Document]:
    """加载 prepare_beir.py 生成的BEIR块文件，包装成Document列表（数据准备步骤）"""
    with open(BEIR_CHUNKS_FILE, encoding="utf-8") as f:
        items = json.load(f)
    docs = [Document(page_content=item["text"], metadata={"chunk": item["id"]})
            for item in items]
    print(f"BEIR模式：已加载 {len(docs)} 个块（来自 {BEIR_CHUNKS_FILE}）")
    return docs


def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """把检索到的片段拼成提示词，调用DeepSeek生成回答（与索引方案无关）"""
    if deepseek_client is None:
        raise RuntimeError(
            "DeepSeek客户端未初始化：请在配置区填写 DEEPSEEK_API_KEY，"
            "或设置环境变量 DEEPSEEK_API_KEY 后重试"
        )
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""你是一位基于给定文档回答问题的助手。请仅根据下面的上下文来回答问题。如果上下文中没有答案，请诚实地说“根据现有文档无法回答该问题”。

上下文：
{context}

问题：{question}
回答："""

    print("正在生成回答...")
    response = deepseek_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content


# --- 主流程 ---
if __name__ == "__main__":
    # 1. 加载索引方案（INDEX_METHOD 决定用哪个脚本，测试对比时改配置即可）
    index_module = load_index_module()
    print(f"当前索引方案：{INDEX_METHOD}")

    # 2. 数据准备：加载 prepare_beir.py 生成的BEIR块文件
    chunks = load_beir_chunks()

    # 3. 构建索引（具体逻辑由索引方案脚本实现）
    #    加 --skip-build 可跳过建库：评测脚本建过库后直接复用，不用再等向量化
    if "--skip-build" in sys.argv:
        print("已指定 --skip-build，跳过建库，直接使用现有索引")
    else:
        index_module.build_index(chunks, INDEX_CONFIG)

    # 4. 检索相关片段（具体逻辑由索引方案脚本实现）
    retrieved_chunks = index_module.search(USER_QUESTION, INDEX_CONFIG)

    # 只检索模式：打印片段后直接结束，方便快速对比不同方案的召回效果
    if RETRIEVE_ONLY:
        print(f"\n===== 检索结果（方案：{INDEX_METHOD}，问题：{USER_QUESTION}）=====")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            print(f"\n--- 片段{i} ---\n{chunk}")
    else:
        # 5. 调用大模型生成最终回答
        answer = generate_answer(USER_QUESTION, retrieved_chunks)
        print(f"\n问题: {USER_QUESTION}\n回答: {answer}")
