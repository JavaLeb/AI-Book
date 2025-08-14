from weaviate.classes.config import Configure, Property, DataType
from weaviate_client import client

with client() as client:
    client.collections.create(
        "Article",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(),
        vector_index_config=Configure.VectorIndex.hnsw(),  # Use the HNSW index
        # vector_index_config=Configure.VectorIndex.flat(),  # Use the FLAT index
        # vector_index_config=Configure.VectorIndex.dynamic(),  # Use the DYNAMIC index
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="body", data_type=DataType.TEXT),
        ]
    )
