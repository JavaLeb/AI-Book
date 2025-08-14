from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate_client import client

with client() as client:
    client.collections.create(
        "Article",
        vectorizer_config=Configure.Vectorizer.text2vec_cohere(),

        properties=[
            Property(
                name="title",
                data_type=DataType.TEXT,
                vectorize_property_name=True,  # Use "title" as part of the value to vectorize
                tokenization=Tokenization.LOWERCASE,  # Use "lowecase" tokenization
                description="The title of the article."  # Optional description
            ),
            Property(
                name="body",
                data_type=DataType.TEXT,
                skip_vectorization=True,  # Don't vectorize this property
                tokenization=Tokenization.WHITESPACE  # Use "whitespace" tokenization
            ),
        ]
    )
