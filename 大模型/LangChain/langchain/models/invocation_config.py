from models import qwen2dot5_7b_model as model
from langchain_core.callbacks import BaseCallbackHandler


class ConfigInspectorHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        # run_name, tags, metadata 会出现在 kwargs 或 config 参数中
        print("--- Chain Start ---")
        print("run_name:", kwargs.get("run_name"))
        print("tags:", kwargs.get("tags"))
        print("metadata:", kwargs.get("metadata"))

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("--- LLM Start ---")
        print("run_name:", kwargs.get("run_name"))
        # 配置信息同样会传递给 LLM 级别的回调
        print("tags:", kwargs.get("tags"))
        print("metadata:", kwargs.get("metadata"))


my_callback_handler = ConfigInspectorHandler()
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",  # Custom name for this run
        "tags": ["humor", "demo"],  # Tags for categorization
        "metadata": {"user_id": "123"},  # Custom metadata
        "callbacks": [my_callback_handler],  # Callback handlers
    }
)

print(response)
