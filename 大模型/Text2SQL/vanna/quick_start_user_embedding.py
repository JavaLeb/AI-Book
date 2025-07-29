from fastembed.common.types import NumpyArray
from typing import Iterable, List
from transformers import AutoModel, AutoTokenizer
import torch


class BCEEmbedding:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def embed(self, texts: List[str]) -> Iterable[NumpyArray]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        yield from outputs.last_hidden_state.mean(dim=1).numpy()


# 初始化自定义模型
embedding_model = BCEEmbedding(model_path=r"E:\LLM\netease-youdao\pytorch\bce-embedding-base_v1")
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

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        # 覆盖默认方法，使用自定义嵌入模型
        embedding = next(self.embedding_model.embed(data))

        return embedding.tolist()  # 转换为 Python 列表


# 初始化Qdrant.
qdrant_client = QdrantClient(url="http://localhost:6333")
try:
    # 初始化Vann.
    vn = MyVanna(config={'client': qdrant_client,
                         'embedding_model': embedding_model,
                         'model': 'llama3.2'})
    # 连接MySQL数据库.
    vn.connect_to_mysql(host='localhost', dbname='sakila', user='root', password='root', port=3306)

    vn.train(ddl="""
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
    # print('training_data=', training_data)

    res = vn.ask(question='请生成纯SQL语句，不要包含任何注释或表格：last_name为WOOD的actor是谁？',
                 allow_llm_to_see_data=True)
    print('res=', res)
finally:
    qdrant_client.close()
