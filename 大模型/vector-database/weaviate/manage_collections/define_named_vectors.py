from weaviate_client import client
from weaviate.classes.config import Configure, Property, DataType

with client() as client:
    client.collections.create(
        "ArticleNV",
        vectorizer_config=[
            # Set a named vector with the "text2vec-cohere" vectorizer
            Configure.NamedVectors.text2vec_cohere(
                name="title",
                source_properties=["title"],  # (Optional) Set the source property(ies)
                vector_index_config=Configure.VectorIndex.hnsw()  # (Optional) Set vector index options
            ),
            # Set another named vector with the "text2vec-openai" vectorizer
            Configure.NamedVectors.text2vec_openai(
                name="title_country",
                source_properties=["title", "country"],  # (Optional) Set the source property(ies)
                vector_index_config=Configure.VectorIndex.hnsw()  # (Optional) Set vector index options
            ),
            # Set a named vector for your own uploaded vectors
            Configure.NamedVectors.none(
                name="custom_vector",
                vector_index_config=Configure.VectorIndex.hnsw()  # (Optional) Set vector index options
            )
        ],
        properties=[  # Define properties
            Property(name="title", data_type=DataType.TEXT),
            Property(name="country", data_type=DataType.TEXT),
        ],
    )
