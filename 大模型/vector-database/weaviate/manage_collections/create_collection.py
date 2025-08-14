from weaviate_client import client

with client() as client:
    client.collections.create("Article")
