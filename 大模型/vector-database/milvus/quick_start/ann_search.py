from pymilvus import MilvusClient


def print_res(res, message):
    print('===========================================')
    print(message)
    for hits in res:
        print('---------------')
        for hit in hits:
            print(hit)


client = MilvusClient()
collection_name = 'data_operation'
partition_name = "partitionA"
# 当向量搜索.
query_vector = [0.4762662351131439, -0.694250226020813]
res = client.search(
    collection_name=collection_name,
    anns_field="vector", data=[query_vector], limit=3,
    search_params={"metric_type": "COSINE"},  # 省略也可以.
)
print_res(res, message="单向量搜索结果:")

# 偏移: limit表示最终要返回的结果数量, offset表示要偏移的数量.
res = client.search(
    collection_name=collection_name,
    anns_field="vector", data=[query_vector], limit=3,
    search_params={"metric_type": "COSINE", "offset": 1},  # 省略也可以.
)
print_res(res, message="单向量搜索结果（偏移）:")

# 多向量搜索.
query_vectors = [
    [0.1459432691335678, -0.3775407373905182],
    [0.06998888403177261, 0.858281672000885]
]
res = client.search(
    collection_name=collection_name,
    data=query_vectors, limit=3, output_fields=["color"])
print_res(res, message="多向量搜索结果:")

# 在分区中搜索.
query_vectors = [
    [0.4762662351131439, -0.694250226020813],
    [0.4452349528804562, -0.8757026943054742]
]
res = client.search(
    collection_name=collection_name,
    partition_names=[partition_name],
    data=[query_vector],
    limit=3,
)
print_res(res, message="分区中搜索结果:")
