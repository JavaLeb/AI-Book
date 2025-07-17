from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

# 创建集合:(1)创建模式, (2) 创建索引参数, (3)创建集合.
# 创建模式: 启动动态字段, 不启动自动id.

# ========================模式========================
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
# 向模式中添加整数类型字段, 主字段.
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)  # 标量字段.
# 向模式中添加浮点向量字段, 向量维度为5.
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)  # 向量字段.

# ========================索引========================
# 创建索引参数.
index_params = client.prepare_index_params()
# 标量字段定义自动索引: INVERTED.
index_params.add_index(field_name="my_id", index_type="AUTOINDEX")
# 向量字段定义自动索引:FLAT, 定义度量类型为余弦相似度.
index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")

# 创建集合.
client.create_collection(
    collection_name="demo",
    schema=schema, index_params=index_params)  # 设置分区数.

# ========================加载与释放========================
# 查看集合的加载状态, {'state': <LoadState: Loaded>}.
load_state = client.get_load_state(collection_name="demo")
print('集合的加载状态:', load_state)
# 释放集合.
client.release_collection(collection_name="demo")
load_state = client.get_load_state(collection_name="demo")
print('集合的加载状态:', load_state)
# 加载集合字段.
client.load_collection(
    collection_name="demo",
    load_fields=["my_id", "my_vector"],
    skip_load_dynamic_field=True)
load_state = client.get_load_state(collection_name="demo")
print('集合字段加载状态:', load_state)

# ========================分区========================
# 查看默认分区.
res = client.list_partitions(collection_name="demo")
print('集合默认分区列表:', res)
# 创建分区.
client.create_partition(collection_name="demo", partition_name="partitionA")
client.create_partition(collection_name="demo", partition_name="partitionB")
# 查看创建后的分区.
res = client.list_partitions(collection_name="demo")
print("创建分区后的集合分区列表:", res)
# 加载分区.
client.load_partitions(collection_name="demo",
                       skip_load_dynamic_field=True, partition_names=["partitionA"])
res = client.get_load_state(collection_name="demo", partition_name="partitionA")
print(res)
# 删除分区, 删除前分区需要被释放.
client.release_partitions(collection_name="demo", partition_names=["partitionA"])
client.drop_partition(collection_name="demo", partition_name="partitionA")
res = client.list_partitions(collection_name="demo")
print('删除分区后, 集合分区列表:', res)

# ========================别名========================
# 创建集合的别名, 别名全局唯一.
client.create_alias(collection_name="demo", alias="bob")
client.create_alias(collection_name="demo", alias="alice")
# 查看集合别名列表.
list_aliases = client.list_aliases(collection_name="demo")
print('集合的别名:', list_aliases)
describe_alias = client.describe_alias(alias="bob")
print('集合别名描述:', describe_alias)

# 创建没有索引的集合.
# 向模式中添加字符串类型字段, 最大长度512.
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR,
                 max_length=512, is_partition_key=True)  # 标量字段.
client.create_collection(
    collection_name="customized_setup_2",
    schema=schema, num_partitions=128)
# {'state': <LoadState: NotLoad>}.
load_state = client.get_load_state(collection_name="customized_setup_2")
print('释放后，集合的加载状态:', load_state)

client.alter_alias(collection_name="customized_setup_2", alias="alice")
res = client.list_aliases(collection_name="customized_setup_2")
print('新集合的别名列表:', res)
res = client.list_aliases(collection_name="demo")
print('原集合的别名列表:', res)
# 删除别名，不用指定集合名称.
client.drop_alias(alias="bob")

# ========================分片========================
# 创建集合:指定分片数.
client.create_collection(collection_name="customized_setup_3", schema=schema, num_shards=1)

# ========================TTL========================
# 创建集合指定TTL.
client.create_collection(
    collection_name="customized_setup_5", schema=schema,
    properties={
        "collection.ttl.seconds": 86400
    }
)
# 对已创建的集合设置TTL.
client.alter_collection_properties(
    collection_name="demo",
    properties={"collection.ttl.seconds": 1209600})
# 删除集合属性TTL.
client.drop_collection_properties(
    collection_name="demo",
    property_keys=["collection.ttl.seconds"]
)

# ========================级别一致性========================
# 创建集合: 设置一致性级别.
client.create_collection(
    collection_name="customized_setup_6",
    schema=schema, consistency_level="Bounded")

# 查看集合列表.
list_collections = client.list_collections()
print('集合名称列表为:', list_collections)

# 描述集合.
describe_collection = client.describe_collection(
    collection_name="demo")
print('集合描述信息:', describe_collection)

# 重命名集合.
client.rename_collection(
    old_name="customized_setup_2",
    new_name="customized_setup_2_new"
)

# 删除集合.
client.drop_collection(
    collection_name="demo"
)
