from weaviate_client import client
from weaviate.classes.config import Configure, Property, DataType

with client() as client:
    client.collections.create(
        "Article",
        # Additional settings not shown
        properties=[  # properties configuration is optional
            Property(
                name="title",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="chunk",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="chunk_number",
                data_type=DataType.INT,
                index_range_filters=True,
            ),
        ],
        inverted_index_config=Configure.inverted_index(  # Optional
            bm25_b=0.7,
            bm25_k1=1.25,
            index_null_state=True,
            index_property_length=True,
            index_timestamps=True
        )
    )
