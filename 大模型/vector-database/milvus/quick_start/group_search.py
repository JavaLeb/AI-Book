from pymilvus import MilvusClient

client = MilvusClient()
query_vector = [0.3580376395471989, -0.6023495712049978]
res = client.search(
    collection_name="data_operation",
    data=[query_vector],
    limit=3,    # 返回的组数.
    group_size=3,   # 每组个数.
    # strict_group_size=True,
    group_by_field="color",
    output_fields=["color"],
    search_params={"metric_type": "COSINE"}
)
for r in res:
    for a in r:
        print(a)
# print(res)
color = [result['entity']['color'] for result in res[0]]
print(color)
