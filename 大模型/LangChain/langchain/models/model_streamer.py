from langchain.chat_models import init_chat_model

model = init_chat_model(model="ollama:qwen2.5:7b")

for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)

print("\n----------")
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text, end="|\n")
print("----------")
if full:
    print(full.content_blocks)
