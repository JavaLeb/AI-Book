from langchain_core.callbacks import UsageMetadataCallbackHandler
from models import qwen2dot5_7b_model as model

callback = UsageMetadataCallbackHandler()
result = model.invoke("Hello", config={"callbacks": [callback]})
print(result)
print(callback.usage_metadata)