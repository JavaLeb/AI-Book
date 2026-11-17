from langchain.chat_models import init_chat_model

model = init_chat_model(model="ollama:qwen2.5:7b")
response = model.invoke(input="Why do parrots talk?")
print(response)
