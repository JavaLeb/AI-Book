from weaviate.classes.config import Configure
from weaviate_client import client

with client() as client:
    client.collections.create(
        "Article",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(),
        reranker_config=Configure.Reranker.cohere()
    )
