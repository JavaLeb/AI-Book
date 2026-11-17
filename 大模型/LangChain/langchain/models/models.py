from langchain.chat_models import init_chat_model

qwen2dot5_7b_model = init_chat_model(model="ollama:qwen2.5:7b")

qwen2dot5_7b_instruct_model = init_chat_model(model="ollama:qwen2.5:7b-instruct")