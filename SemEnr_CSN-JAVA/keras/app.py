# app.py
# -*- coding: utf-8 -*-

import numpy as np
import gradio as gr

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import configs
from configs import get_config
from models import *  # 确保能拿到你的 JointEmbeddingModel 等
from main import CodeSearcher, configure_gpu_memory_growth  # 复用你已有的类和GPU配置

# -----------------------------
# 1. GPU 设置（复用你在 main.py 里的函数）
# -----------------------------
configure_gpu_memory_growth()

# -----------------------------
# 2. 全局初始化：config, CodeSearcher, model, 数据
# -----------------------------

# 2.1 读取配置（等价于 main.py 的 getattr(configs, args.proto)()）
conf = get_config()  # 如果你有多个 config，可以按需改

# 2.2 构建 CodeSearcher
codesearcher = CodeSearcher(conf)

# 2.3 构建模型（与 main.py 完全一致）
print("Building model...")
model_cls_name = conf["model_params"]["model_name"]   # e.g. "JointEmbeddingModel"
ModelClass = eval(model_cls_name)
model = ModelClass(conf)
model.build()
optimizer = Adam(clipnorm=0.1)
model.compile(optimizer=optimizer)

# 2.4 加载训练好的权重（依赖 training_params['reload']）
reload_epoch = conf["training_params"].get("reload", 0)
if reload_epoch > 0:
    print(f"Loading trained weights from epoch {reload_epoch} ...")
    codesearcher.load_model_epoch(model, reload_epoch)
else:
    print("WARNING: training_params['reload'] == 0，当前模型可能是随机初始化权重！")

# 2.5 预加载 codebase（文本代码库）
codesearcher.load_codebase()  # 会填充 codesearcher._code_base

# 2.6 预加载用来做搜索的向量数据（tokens + sim_desc）
print("Loading use_data (tokens + sim_desc) ...")
use_tokens, use_sim_desc = codesearcher.load_use_data()

# 2.7 统一 padding，避免每次请求重复 pad
tokens_len = conf["data_params"]["tokens_len"]
sim_desc_len = conf["data_params"]["sim_desc_len"]
desc_len = conf["data_params"]["desc_len"]

padded_tokens = codesearcher.pad(use_tokens, tokens_len)
padded_sim_desc = codesearcher.pad(use_sim_desc, sim_desc_len)
data_len = len(use_tokens)

print(f"Search pool size: {data_len} code snippets.")


# -----------------------------
# 3. 封装一个“查询函数”供 Gradio 调用
# -----------------------------
def search_code(query: str, top_k: int = 5):
    """
    输入自然语言描述 (query)，返回 top_k 条代码及相似度。
    逻辑基本复用 CodeSearcher.search()，只是不再每次重新 load 数据。
    """

    if not query.strip():
        return "请输入一个非空的查询。"

    # 3.1 query -> 词表 index
    desc_indices = codesearcher.convert(codesearcher.vocab_desc, query)
    padded_desc = codesearcher.pad([desc_indices] * data_len, desc_len)

    # 3.2 调用模型做相似度预测
    sims = model.predict(
        [padded_tokens, padded_sim_desc, padded_desc],
        batch_size=1000,
        verbose=0
    ).flatten()  # shape: (data_len,)

    # 3.3 取 Top-K
    if top_k > data_len:
        top_k = data_len

    negsims = np.negative(sims)
    # 先选出 top_k 个索引
    candidate_inds = np.argpartition(negsims, kth=top_k - 1)[:top_k]
    # 再按相似度从大到小排序
    candidate_sims = sims[candidate_inds]
    order = np.argsort(-candidate_sims)  # 降序
    top_inds = candidate_inds[order]
    top_sims = candidate_sims[order]

    # 3.4 组装成可读字符串
    outputs = []
    for rank, (idx, score) in enumerate(zip(top_inds, top_sims), 1):
        code_snippet = codesearcher._code_base[idx]
        outputs.append(f"【Top {rank} | score={score:.4f}】\n{code_snippet}")

    return "\n\n" + ("-" * 60 + "\n\n").join(outputs)


# -----------------------------
# 4. Gradio “聊天界面”封装
# -----------------------------

def chat_fn(message, history):
    """
    Gradio ChatInterface 回调函数。
    - message: 当前这一条用户输入
    - history: 之前的 (user, bot) 列表，这里我们完全不使用，实现“单轮问答”。
    """
    # 单轮：只看当前 message
    result_text = search_code(message, top_k=5)
    return result_text


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Code Search Demo (自训练模型)",
    description=(
        "输入自然语言描述（英文或你训练时使用的语言），"
        "模型会在代码库中检索相似的代码片段，返回 Top-K 结果。\n"
        "逻辑上是单轮问答，不记忆历史对话，只是为了界面用 Chat 样式。"
    ),
    examples=[
        ["convert an inputstream to a string"],
        ["read a file line by line in Java"],
        ["sort a list using comparator"]
    ]
)

if __name__ == "__main__":
    # server_name="0.0.0.0" 方便在局域网/外部访问
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
