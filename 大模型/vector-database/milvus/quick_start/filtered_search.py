from pymilvus import MilvusClient

client = MilvusClient()
collection_name = 'data_operation'
# 通过设置ttl删除所有数据.
client.delete(collection_name=collection_name, filter="id >= 0")

data = [
    {"id": 0, "vector": [0.39524717779832685, 0.4000257286739164], "color": "pink"},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088], "color": "red"},
    {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526], "color": "orange"},
    {"id": 3, "vector": [0.3580376395471989, -0.6023495712049978], "color": "pink"},
    {"id": 4, "vector": [-0.33445148015177995, -0.2567135004164067], "color": "red"},
    {"id": 5, "vector": [0.985825131989184, -0.8144651566660419], "color": "yellow"},
    {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833], "color": "red"},
    {"id": 7, "vector": [0.5718280481994695, 0.24070317428066512], "color": "grey"},
    {"id": 8, "vector": [0.3580376395471989, -0.6023495712049978], "color": "white"},
    {"id": 9, "vector": [0.4452349528804562, -0.8757026943054742], "color": "pink"}
]
client.insert(collection_name=collection_name, data=data)

query_vector = [0.3580376395471989, -0.6023495712049978]
res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=2,
    filter='color like "red%"',
    output_fields=["color"]
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)

res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=2,
    filter='color like "red%"',
    output_fields=["color"],
    search_params={"hints": "iterative_filter"}
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
