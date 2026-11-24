from models import qwen2dot5_7b_model as model

tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])
response = model_with_tools.invoke("What was a positive news story from today?")
print(response.content_blocks)
