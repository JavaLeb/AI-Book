from abc import ABC

from fastembed.text.text_embedding import OnnxTextEmbedding
from typing import Any
from fastembed.common.model_description import DenseModelDescription, ModelSource


class BCEEmbedding(OnnxTextEmbedding, ABC):
    def __init__(self, **kwargs: Any):
        from fastembed.text.onnx_embedding import supported_onnx_models

        bce = DenseModelDescription(
            model="netease-youdao/bce-embedding-base_v1",
            dim=768,
            description="Text embeddings",
            license="apache-2.0",
            size_in_GB=0.52,
            sources=ModelSource(hf="netease-youdao/bce-embedding-base_v1"),
            model_file="embed.onnx",
        )
        supported_onnx_models.append(bce)

        super().__init__(**kwargs)


# 初始化
embedding_model = BCEEmbedding(
    model_name="netease-youdao/bce-embedding-base_v1",  # 模型名称, Pytorch格式.
    cache_dir=r"E:\LLM\netease-youdao\onnx",
    local_files_only=True
)


embeddings = list(embedding_model.embed(["Hello, world!"]))
print('embedding=', embeddings[0].shape)  # 应输出 384 维向量

from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient


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
