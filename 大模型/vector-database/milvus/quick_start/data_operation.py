from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

# 创建集合.
schema = client.create_schema()
schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=2)
schema.add_field(field_name='color', datatype=DataType.VARCHAR, index_type="AUTOINDEX", max_length=512)
index_params = client.prepare_index_params()
index_params.add_index(field_name="color", index_type="AUTOINDEX")
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
client.create_collection(collection_name='data_operation', schema=schema, index_params=index_params)

# 插入数据.
data = [
    {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978], "color": "pink_8682"},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088], "color": "red_7025"},
    {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526], "color": "orange_6781"},
    {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428], "color": "pink_9298"},

]
res = client.insert(collection_name="data_operation", data=data)
print('插入数据结果:', res)

# 向集合分区插入数据.
data = [
    {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742], "color": "red_4794"},
    {"id": 5, "vector": [0.985825131989184, -0.8144651566660419], "color": "yellow_4222"},
    {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833], "color": "red_9392"}
]
client.create_partition(collection_name="data_operation", partition_name="partitionA")
res = client.insert(collection_name="data_operation",
                    partition_name="partitionA", data=data)
print('向集合分区插入数据结果:', res)

# 插入更新数据.
data = [
    {"id": 0, "vector": [-0.619954382375778, 0.4479436794798608], "color": "black_9898"},
    {"id": 1, "vector": [0.4762662251462588, -0.6942502138717026], "color": "red_7319"},
    {"id": 2, "vector": [-0.8864122635045097, 0.9260170474445351], "color": "white_6465"},
    {"id": 3, "vector": [0.14594326235891586, -0.3775407299900644], "color": "orange_7580"}
]
res = client.upsert(collection_name='data_operation', data=data)
print('插入更新数据结果:', res)
data = [
    {"id": 4, "vector": [0.06998888224297328, 0.8582816610326578], "color": "black_3651"},
    {"id": 7, "vector": [0.6060703043917468, -0.3765080534566074], "color": "grey_2049"}
]
res = client.upsert(collection_name="data_operation",
                    data=data, partition_name="partitionA")
print('向集合分区插入更新数据结果:', res)

# 通过条件过滤删除数据.
res = client.delete(collection_name="data_operation",
                    filter="color in ['red_7025', 'red_9392']")
print('条件删除数据结果:', res)
# 指定主键删除数据.
res = client.delete(collection_name="data_operation",
                    ids=[0, 2])
print('主键删除数据结果:', res)
# 从指定分区中删除数据.
res = client.delete(
    collection_name="data_operation",
    ids=[5, 6], partition_name="partitionA")
print('指定分区删除数据结果:', res)
