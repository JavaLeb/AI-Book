from autogen_ext.models.ollama import OllamaChatCompletionClient

# 模型客户端.
model_client = OllamaChatCompletionClient(
    base_url="http://localhost:11434/v1",
    model="qwen3:1.7b",
)

multi_model_client = OllamaChatCompletionClient(
    base_url="http://localhost:11434/v1",
    model="qwen2.5vl:3b",
)

if __name__ == '__main__':
    print(model_client)
