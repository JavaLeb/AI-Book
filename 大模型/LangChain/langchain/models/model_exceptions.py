from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelNotFoundError, ModelTimeoutError, ModelConnectionError

# 关键修改：使用一个不存在的模型名
model = init_chat_model("ollama:qwen2.5:7b1")

try:
    response = model.invoke("Hello")
    print(response)
except ModelTimeoutError:
    print("caught: ModelTimeoutError")
except ModelNotFoundError:
    print("caught: ModelNotFoundError")
except ModelConnectionError:
    print("caught: ModelConnectionError")
except Exception as e:
    # Ollama 对未知模型通常抛出 ollama._types.ResponseError (status_code=404)
    print(f"caught: {type(e).__name__} | status_code={getattr(e, 'status_code', None)}")
    print(f"原始信息: {e}")