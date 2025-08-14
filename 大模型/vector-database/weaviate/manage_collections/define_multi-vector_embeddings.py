from weaviate_client import client
from weaviate.classes.config import Configure, Property, DataType

with client() as client:
    client.collections.create(
        "DemoCollection",
        vectorizer_config=[
            # Example 1 - Use a model integration
            # The factory function will automatically enable multi-vector support for the HNSW index
            Configure.NamedVectors.text2colbert_jinaai(
                name="jina_colbert",
                source_properties=["text"],
            ),
            # Example 2 - User-provided multi-vector representations
            # Must explicitly enable multi-vector support for the HNSW index
            Configure.NamedVectors.none(
                name="custom_multi_vector",
                vector_index_config=Configure.VectorIndex.hnsw(
                    multi_vector=Configure.VectorIndex.MultiVector.multi_vector()
                ),
            ),
        ],
        properties=[
            Property(name="text", data_type=DataType.TEXT)
        ]
        # Additional parameters not shown
    )