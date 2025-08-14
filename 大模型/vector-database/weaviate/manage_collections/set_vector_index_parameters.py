from weaviate.classes.config import Configure, VectorDistances, VectorFilterStrategy
from weaviate_client import client

with client() as client:
    client.collections.create(
        "Article",
        # Additional configuration not shown
        vector_index_config=Configure.VectorIndex.hnsw(
            quantizer=Configure.VectorIndex.Quantizer.bq(),
            ef_construction=300,
            distance_metric=VectorDistances.COSINE,
            filter_strategy=VectorFilterStrategy.SWEEPING  # or ACORN (Available from Weaviate v1.27.0)
        ),
    )
