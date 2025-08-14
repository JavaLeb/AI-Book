import weaviate
import json
from weaviate.classes.init import Auth

client = weaviate.connect_to_local(
    host="localhost", port=8080, auth_credentials=Auth.api_key("123456")
)

questions = client.collections.get("Question")

response = questions.query.near_text(
    query="biology",
    limit=2
)

for obj in response.objects:
    print(json.dumps(obj.properties, indent=2))

client.close()  # Free up resources
