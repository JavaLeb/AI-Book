import weaviate
from weaviate.classes.config import Configure

from weaviate.classes.init import Auth

client = weaviate.connect_to_local(
    host="localhost", port=8080, auth_credentials=Auth.api_key("123456")
)
print(client.is_ready())  # Should print: `True`

# 删除集合.
client.collections.delete('Question')
# 使用host.docker.internal的原因: weaviate部署在docker中，ollama部署在宿主机上.
ollama_url = "http://host.docker.internal:11434"
questions = client.collections.create(
    name="Question",
    # 向量化配置.
    vectorizer_config=Configure.Vectorizer.text2vec_ollama(  # Configure the Ollama embedding integration
        api_endpoint=ollama_url,
        model="nomic-embed-text",  # The model to use
    ),
    # 生成配置.
    generative_config=Configure.Generative.ollama(  # Configure the Ollama generative integration
        api_endpoint=ollama_url,
        # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="llama3.2",  # The model to use
    )
)

client.close()  # Free up resources
