from weaviate_client import client
from weaviate.classes.config import Property, DataType

with client() as client:
    articles = client.collections.get("Article")

    articles.config.add_property(
        Property(
            name="onHomepage",
            data_type=DataType.BOOL
        )
    )
