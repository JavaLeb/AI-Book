from weaviate_client import client
from weaviate.classes.config import Reconfigure, VectorFilterStrategy, ReplicationDeletionStrategy

with client() as client:

    articles = client.collections.get("Article")

    # Update the collection definition
    articles.config.update(
        description="An updated collection description.",
        property_descriptions={
            "title": "The updated title description for article",
        },  # Available from Weaviate v1.31.0
        inverted_index_config=Reconfigure.inverted_index(
            bm25_k1=1.5
        ),
        vector_index_config=Reconfigure.VectorIndex.hnsw(
            filter_strategy=VectorFilterStrategy.ACORN  # Available from Weaviate v1.27.0
        ),
        replication_config=Reconfigure.replication(
            deletion_strategy=ReplicationDeletionStrategy.TIME_BASED_RESOLUTION  # Available from Weaviate v1.28.0
        )
    )
    articles = client.collections.get("Article")

    article_shards = articles.config.update_shards(
        status="READY",
        shard_names='shard_name'  # The names (List[str]) of the shard to update (or a shard name)
    )

    print(article_shards)