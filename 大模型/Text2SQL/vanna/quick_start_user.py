from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore
from fastembed.text import TextEmbedding
from qdrant_client import QdrantClient
import sys

print("Python 版本:", sys.version)  # 打印详细的版本信息
# 打印支持的模型信息.
models = TextEmbedding.list_supported_models()
for model in models:
    print(model)


class MyVanna(Qdrant_VectorStore, Ollama):
    """
    大模型使用Ollama供应商提供的模型.
    向量数据库使用Qdrant.
    """

    def __init__(self, config=None):
        # 确保 config 包含自定义的嵌入模型
        if config is None:
            config = {}
        if 'embedding_model' not in config:
            raise ValueError("必须提供 embedding_model 参数")
        self.embedding_model = config['embedding_model']

        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

    def generate_embedding(self, data: str, **kwargs) -> list[float]:
        # 覆盖默认方法，使用自定义嵌入模型
        embedding = next(self.embedding_model.embed(data))

        return embedding.tolist()  # 转换为 Python 列表


# 初始化 fastembed 并指定本地模型路径
embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # 模型名称, Pytorch格式.
    cache_dir=r"E:\LLM\qdrant",  # hf格式. 'model': 'BAAI/bge-base-en-v1.5' -> 'hf': 'qdrant/bge-base-en-v1.5-onnx-q'
    local_files_only=True)

embeddings = list(embedding_model.embed(["Hello, world!"]))
print('embedding=', embeddings[0].shape)  # 应输出 384 维向量
# 初始化Qdrant.
qdrant_client = QdrantClient(url="http://localhost:6333")

# 初始化Vann.
vn = MyVanna(config={'client': qdrant_client,
                     'embedding_model': embedding_model,
                     'model': 'llama3.2'})
# 连接MySQL数据库.
vn.connect_to_mysql(host='localhost', dbname='sakila', user='root', password='root', port=3306)

vn.train(ddl="""
-- sakila.actor definition

CREATE TABLE `actor` (
  `actor_id` smallint unsigned NOT NULL AUTO_INCREMENT,
  `first_name` varchar(45) NOT NULL,
  `last_name` varchar(45) NOT NULL,
  `last_update` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`actor_id`),
  KEY `idx_actor_last_name` (`last_name`)
) ENGINE=InnoDB AUTO_INCREMENT=201 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
""")

training_data = vn.get_training_data()
print('training_data=', training_data)

res = vn.ask(question='last_name为WOOD的actor是谁？', allow_llm_to_see_data=True)
print('res=', res)
