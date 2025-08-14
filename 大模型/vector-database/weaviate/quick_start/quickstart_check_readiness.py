import weaviate
from weaviate.classes.init import Auth

client = weaviate.connect_to_local(
    host="localhost", port=8080, auth_credentials=Auth.api_key("123456")
)
print(client.is_ready())  # Should print: `True`
client.close()  # Free up resources
