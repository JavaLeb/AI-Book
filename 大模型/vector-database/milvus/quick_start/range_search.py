from pymilvus import MilvusClient

client = MilvusClient()

query_vector = [0.3580376395471989, -0.6023495712049978]
# radius内圆半径, range_filter外圆半径.
res = client.search(
    collection_name="data_operation",
    data=[query_vector], limit=3,
    output_fields=['color'],
    search_params={
        "metric_type": "COSINE",
        "params": {"radius": 0.4, "range_filter": 0.6}
    }
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
