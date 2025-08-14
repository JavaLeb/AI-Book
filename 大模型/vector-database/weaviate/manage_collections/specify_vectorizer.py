from weaviate_client import client
from weaviate.classes.config import Configure, Property, DataType

with client() as client:
    result = client.collections.create(
        "Article",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(),
        properties=[  # properties configuration is optional
            Property(name="title", data_type=DataType.TEXT),
            Property(name="body", data_type=DataType.TEXT),
        ]
    )
    print(result)  # 返回集合的schema.
