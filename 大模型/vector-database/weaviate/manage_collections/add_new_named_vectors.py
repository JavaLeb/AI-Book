from weaviate_client import client
from weaviate.classes.config import Configure

with client() as client:
    articles = client.collections.get("Article")

    articles.config.add_vector(
        vector_config=Configure.NamedVectors.text2vec_cohere(
            name="body_vector",
            source_properties=["body"],
        )
    )
