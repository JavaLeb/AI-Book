from weaviate_client import client
from weaviate.classes.config import Property, DataType

with client() as client:
    client.collections.create(
        "Article",
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="body", data_type=DataType.TEXT),
        ]
    )