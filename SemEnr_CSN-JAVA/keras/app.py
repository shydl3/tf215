# app.py
# -*- coding: utf-8 -*-

import numpy as np
import gradio as gr
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import configs
from configs import get_config
from models import *  
from main import CodeSearcher, configure_gpu_memory_growth
 

# 加载参数，模型
configure_gpu_memory_growth()
conf = get_config()  
codesearcher = CodeSearcher(conf)

print("Building model...")
model_cls_name = conf["model_params"]["model_name"]   # e.g. "JointEmbeddingModel"
ModelClass = eval(model_cls_name)
model = ModelClass(conf)
model.build()
optimizer = Adam(clipnorm=0.1)
model.compile(optimizer=optimizer)

# reload_epoch = conf["training_params"]["reload"]
# if reload_epoch > 0:
#     print(f"Loading trained weights from epoch {reload_epoch} ...")
#     codesearcher.load_model_epoch(model, reload_epoch)
# else:
#     print("WARNING: training_params['reload'] == 0，随机初始化权重！")


if conf['training_params']['reload'] > 0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
else:
    print("WARNING: training_params['reload'] == 0，随机初始化权重！")

codesearcher.load_codebase()  

print("Loading use_data (tokens + sim_desc) ...")
use_tokens, use_sim_desc = codesearcher.load_use_data()

tokens_len = conf["data_params"]["tokens_len"]
sim_desc_len = conf["data_params"]["sim_desc_len"]
desc_len = conf["data_params"]["desc_len"]

padded_tokens = codesearcher.pad(use_tokens, tokens_len)
padded_sim_desc = codesearcher.pad(use_sim_desc, sim_desc_len)
data_len = len(use_tokens)

print(f"Search pool size: {data_len} code snippets.")

def search_code(query: str, top_k: int = 5):
    codes, sims = codesearcher.search(model, query, n_results=top_k)

    # 如果你想稳妥一点，可以再按 sims 降序排一下
    pairs = list(zip(codes, sims))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_k]

    blocks = []
    for rank, (code_snippet, score) in enumerate(pairs, 1):
        formatted = pretty_format_java(code_snippet)
        block = f"""**Top {rank}**  
score = `{score:.4f}`

```java
{formatted}
```"""
        blocks.append(block)

    return "\n\n".join(blocks)


# def search_code(query: str, top_k: int = 5):
#     """
#     输入自然语言描述 (query)，返回 top_k 条代码及相似度。
#     逻辑基本复用 CodeSearcher.search()，只是不再每次重新 load 数据。
#     """

#     if not query.strip():
#         return "请输入要查询的代码功能描述。"

#     # print(f"{query}")
#     # sys.exit(1)
    
#     print("RAW QUERY:", repr(query))

#     # 看看分词和索引
#     tokens = query.strip().lower().split(' ')
#     desc_indices = codesearcher.convert(codesearcher.vocab_desc, query)
#     print("TOKENS:", tokens)
#     print("INDICES:", desc_indices)


#     # 3.3 取 Top-K
#     if top_k > data_len:
#         top_k = data_len
    
#     n_results = top_k


#     codes, sims = codesearcher.search(model, query, top_k)
#     zipped = list(zip(codes, sims))
#     zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
#     zipped = codesearcher.postproc(zipped)
#     zipped = list(zipped)[:n_results]
#     results = '\n\n'.join(map(str, zipped))  # combine the result into a returning string


#     negsims = np.negative(sims)
#     # 先选出 top_k 个索引
#     candidate_inds = np.argpartition(negsims, kth=top_k - 1)[:top_k]
#     # 再按相似度从大到小排序
#     candidate_sims = sims[candidate_inds]
#     order = np.argsort(-candidate_sims)
#     top_inds = candidate_inds[order]
#     top_sims = candidate_sims[order]

#     outputs = []
#     for rank, (idx, score) in enumerate(zip(top_inds, top_sims), 1):
#         code_snippet = codesearcher._code_base[idx]

#         # 用 Markdown 代码块包装
#         block = f"""**Top {rank}**  \nscore = `{score:.4f}`
# ```java
# {code_snippet}
# ```"""
#         outputs.append(block)

#     return "\n\n".join(outputs)


def pretty_format_java(code: str) -> str:
    code = code.replace("{", "{\n")
    code = code.replace("}", "\n}\n")
    code = code.replace(";", ";\n")

    lines = [l.strip() for l in code.split("\n") if l.strip()]

    indent = 0
    new_lines = []
    for line in lines:
        if line.startswith("}"):
            indent -= 1
        new_lines.append("    " * indent + line)
        if line.endswith("{"):
            indent += 1
    return "\n".join(new_lines)


def chat_fn(message, history):
    """
    Gradio ChatInterface 回调函数。
    - message: 当前这一条用户输入
    - history: 之前的 (user, bot) 列表,实现单轮问答。
    """
    result_text = search_code(message, top_k=5)

    format_text = pretty_format_java(result_text)

    # print(format_text)
    # sys.exit(1)

    # return result_text
    return format_text


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Code Search Demo (Based on CSN-JAVA)",
    description=(
        "请输入自然语言描述，"
        "模型会在代码库中检索相似的代码片段，返回 Top-K 结果。\n"
        "逻辑上是单轮问答，不记忆历史对话。"
    ),
    examples=[
        ["convert an inputstream to a string"],
        ["read a file line by line in Java"],
        ["sort a list using comparator"]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
