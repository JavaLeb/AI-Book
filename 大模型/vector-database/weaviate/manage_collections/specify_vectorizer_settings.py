from weaviate_client import client
from weaviate.classes.config import Configure

with client() as client:
    client.collections.create(
        "Article",
        vectorizer_config=Configure.Vectorizer.text2vec_cohere(
            model="embed-multilingual-v2.0",
            vectorize_collection_name=True
        ),
    )
