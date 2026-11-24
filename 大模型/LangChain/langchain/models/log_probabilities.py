from models import qwen2dot5_7b_model as model


model = model.bind(logprobs=True)

response = model.invoke("Why do parrots talk?")
print(response.response_metadata["logprobs"])