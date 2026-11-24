from models import qwen2dot5_7b_model as model
from langchain.chat_models import init_chat_model

import langchain_core
print("langchain-core:", langchain_core.__version__)

print("has profile attr:", hasattr(model, "profile"))
print("profile repr:", repr(getattr(model, "profile", "<no attr>")))
print("model type:", type(model))
print("model name:", getattr(model, "model_name", None) or getattr(model, "model", None))

print(model.profile)

custom_profile = {
    "max_input_tokens": 100_000,
    "tool_calling": True,
    "structured_output": True,
}
model = init_chat_model("ollama:qwen2.5:7b", profile=custom_profile)
print(model.profile)