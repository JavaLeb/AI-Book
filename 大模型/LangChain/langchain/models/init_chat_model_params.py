from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="ollama:qwen2.5:7b",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,  # Default; increase for unreliable networks
)
response = model.invoke(input="Why do parrots talk?")
print(response)
