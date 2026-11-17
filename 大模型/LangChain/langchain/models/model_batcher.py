from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

model = init_chat_model(model="ollama:qwen2.5:7b")

responses = model.batch(inputs=[
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
],config=RunnableConfig(max_concurrency=5))
for response in responses:
    print(response)

for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)
