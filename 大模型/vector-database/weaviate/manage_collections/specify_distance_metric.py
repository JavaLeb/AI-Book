from weaviate.classes.config import Configure, VectorDistances
from weaviate_client import client

with client() as client:
    client.collections.create(
        "Article",
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        ),
    )
