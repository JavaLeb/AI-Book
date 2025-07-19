from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient()

schema = client.create_schema()
# 添加主键字段, 自动ID.
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
# 添加文本字段, 启动分析器, 默认使用标准分析器, 分析器用于分词和过滤等.
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True)
# 添加稀疏向量字段.
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

# 稀疏向量字段添加索引, 在构建索引时, 计算相关性使用BM25.
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",  # 构建和查询索引算法.
        "bm25_k1": 1.2, "bm25_b": 0.75  # BM25算法超参数.
    }
)

# bm25函数: 将文本字段转换为稀疏向量.
bm25_function = Function(
    name="text_bm25_emb",  # 函数名: 自定义即可.
    input_field_names=["text"],  # 需要转换为稀疏向量的文本字段.
    output_field_names=["sparse"],  # 存储转化后稀疏向量的字段.
    function_type=FunctionType.BM25
)
schema.add_function(bm25_function)

# 文档.
docs = [
    {'text': 'information retrieval is a field of study.'},
    {'text': 'information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'data mining and information retrieval overlap in research.'},
]
collection_name = 'my_collection'
# 创建集合.
client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
# 写入数据.
client.delete(collection_name=collection_name, filter='id >= 0')
client.insert(collection_name, data=docs)

# 全文搜索.
query = 'whats the focus of information retrieval?'
res = client.search(
    collection_name=collection_name,
    data=[query],
    anns_field='sparse', output_fields=['text'], limit=3,
    search_params={'params': {'drop_ratio_search': 0.2}}  # 忽略低重要性术语比例.
)
# 输出全文搜索结果.
for records in res:
    for record in records:
        print(record)
