from langchain.chat_models import init_chat_model

qwen2dot5_7b_model = init_chat_model(model="ollama:qwen2.5:7b")
qwen2dot5_7b_instruct_model = init_chat_model(model="ollama:qwen2.5:7b-instruct")

# 图 ->  文.
# 不支持 文 -> 图.
qwen3_VL_2b_model = init_chat_model(model="ollama:Qwen3-VL:2b")
qwen3_VL_4b_model = init_chat_model(model="ollama:Qwen3-VL:4b")
