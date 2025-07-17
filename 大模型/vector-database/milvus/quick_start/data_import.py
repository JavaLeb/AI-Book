from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType, RemoteBulkWriter
from pymilvus.bulk_writer import bulk_import, get_import_progress
import random, string, json
import numpy as np

# 源数据中包含了id和多余的字段, 因此不启动自动id,启动动态字段.
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
DIM = 512
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="bool", datatype=DataType.BOOL)
schema.add_field(field_name="int8", datatype=DataType.INT8)
schema.add_field(field_name="int16", datatype=DataType.INT16)
schema.add_field(field_name="int32", datatype=DataType.INT32)
schema.add_field(field_name="int64", datatype=DataType.INT64)
schema.add_field(field_name="float", datatype=DataType.FLOAT)
schema.add_field(field_name="double", datatype=DataType.DOUBLE)
schema.add_field(field_name="varchar", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="json", datatype=DataType.JSON)
schema.add_field(field_name="array_str", datatype=DataType.ARRAY,
                 max_capacity=100, element_type=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="array_int", datatype=DataType.ARRAY,
                 max_capacity=100, element_type=DataType.INT64)
schema.add_field(field_name="float_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field(field_name="binary_vector", datatype=DataType.BINARY_VECTOR, dim=DIM)
schema.add_field(field_name="float16_vector", datatype=DataType.FLOAT16_VECTOR, dim=DIM)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.verify()
print('集合模式:', schema)

local_bulk_writer = LocalBulkWriter(
    schema=schema,  # 文件中数据的schema.
    local_path='.',  # 输出目录.
    segment_size=512 * 1024 * 1024,  # 段大小, 根据导入的数据来设置.
    file_type=BulkFileType.PARQUET  # 输出文件格式: PARQUET是一种列式存储格式.
)

ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "a-bucket"
# 通过S3协议连接Milvus内置的MinIO.
conn = RemoteBulkWriter.S3ConnectParam(
    endpoint="localhost:9000",  # the default MinIO service started along with Milvus
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    bucket_name=BUCKET_NAME,
    secure=False
)
remote_bulk_writer = RemoteBulkWriter(
    schema=schema,
    remote_path="/",
    connect_param=conn,
    file_type=BulkFileType.PARQUET
)
print('bulk writer created.')


def generate_random_str(length=5):
    letters = string.ascii_uppercase
    digits = string.digits

    return ''.join(random.choices(letters + digits, k=length))


def gen_binary_vector(to_numpy_arr):
    raw_vector = [random.randint(0, 1) for i in range(DIM)]
    if to_numpy_arr:
        return np.packbits(raw_vector, axis=-1)
    return raw_vector


def gen_float_vector(to_numpy_arr):
    raw_vector = [random.random() for _ in range(DIM)]
    if to_numpy_arr:
        return np.array(raw_vector, dtype="float32")
    return raw_vector


def gen_fp16_vector(to_numpy_arr):
    raw_vector = [random.random() for _ in range(DIM)]
    if to_numpy_arr:
        return np.array(raw_vector, dtype=np.float16)
    return raw_vector


def gen_sparse_vector(pair_dict: bool):
    raw_vector = {}
    dim = random.randint(2, 20)
    if pair_dict:
        raw_vector["indices"] = [i for i in range(dim)]
        raw_vector["values"] = [random.random() for _ in range(dim)]
    else:
        for i in range(dim):
            raw_vector[i] = random.random()
    return raw_vector


# 使用remote_bulk_writer
writer = remote_bulk_writer
for i in range(10000):
    writer.append_row({
        "id": np.int64(i),
        "bool": True if i % 3 == 0 else False,
        "int8": np.int8(i % 128),
        "int16": np.int16(i % 1000),
        "int32": np.int32(i % 100000),
        "int64": np.int64(i),
        "float": np.float32(i / 3),
        "double": np.float64(i / 7),
        "varchar": f"varchar_{i}",
        "json": json.dumps({"dummy": i, "ok": f"name_{i}"}),
        "array_str": np.array([f"str_{k}" for k in range(5)], np.dtype("str")),
        "array_int": np.array([k for k in range(10)], np.dtype("int64")),
        "float_vector": gen_float_vector(True),
        "binary_vector": gen_binary_vector(True),
        "float16_vector": gen_fp16_vector(True),
        "sparse_vector": gen_sparse_vector(True),
        f"dynamic_{i}": i,
    })
    if (i + 1) % 1000 == 0:  # 提交数据.
        writer.commit()
        print('committed')

print(writer.batch_files)

client = MilvusClient()
# 创建集合, 向量字段需要设置索引.
index_params = client.prepare_index_params()
index_params.add_index(field_name="float_vector", index_type="AUTOINDEX")
index_params.add_index(field_name="binary_vector", index_type="AUTOINDEX")
index_params.add_index(field_name="float16_vector", index_type="AUTOINDEX")
index_params.add_index(field_name="sparse_vector", index_type="AUTOINDEX")
client.create_collection(collection_name='data_import',
                         schema=schema, index_params=index_params)
# 从MinIO导入集合.
url = "http://localhost:19530"
resp = bulk_import(
    url=url, collection_name="data_import",
    files=[['\\1.parquet'], ['\\2.parquet'],
           ['\\3.parquet'], ['\\4.parquet'],
           ['\\5.parquet'], ['\\6.parquet'],
           ['\\7.parquet'], ['\\8.parquet'],
           ['\\9.parquet'], ['\\10.parquet']])
# 获取任务id.
job_id = resp.json()['data']['jobId']
print('import job id:', job_id)

# 获取导入进度.
resp = get_import_progress(
    url=url,
    job_id=job_id,
)
print(json.dumps(resp.json(), indent=4))
