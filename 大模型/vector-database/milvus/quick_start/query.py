from pymilvus import MilvusClient
from pymilvus import connections, Collection

client = MilvusClient()
collection_name = 'data_operation'
entities = client.get(
    collection_name=collection_name,
    ids=[0, 1, 2],
    output_fields=["vector", "color"]
)
print('Get Results:')
for entity in entities:
    print(entity)

entities = client.query(
    collection_name=collection_name,
    filter="color like \"red%\"",
    output_fields=["vector", "color"],
    limit=3
)
print('Query Results:')
for entity in entities:
    print(entity)

connections.connect()
collection = Collection(collection_name)
iterator = collection.query_iterator(
    batch_size=1,  # 批次大小.
    expr="color like \"red%\"",
    output_fields=["color"],
    limit=3  # 返回的记录数.
)
results = []
print('QueryIterator Results:')
while True:
    result = iterator.next()
    if not result:
        iterator.close()
        break
    print(result)
    results += result
