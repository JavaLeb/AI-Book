from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

model = init_chat_model(
    model="llama3.2",
    base_url='http://localhost:11434/',
    model_provider="ollama")

print('模型初始化完成')
