from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434",
    api_key="YOUR_API_KEY",
)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-6.1-sol",
    openai_proxy="http://proxy.example.com:8080"
)