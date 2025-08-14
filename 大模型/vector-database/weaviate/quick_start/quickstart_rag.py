import weaviate
from weaviate.classes.init import Auth

client = weaviate.connect_to_local(
    host="localhost", port=8080, auth_credentials=Auth.api_key("123456")
)
questions = client.collections.get("Question")
response = questions.generate.near_text(
    query="biology",
    limit=2,
    grouped_task="用表情符号写一条关于这些事实的推特。"
)
print(response.generated)  # Inspect the generated text
client.close()
