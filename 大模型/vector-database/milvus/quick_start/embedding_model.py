from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.sparse import SpladeEmbeddingFunction

from pymilvus import model
from modelscope import snapshot_download

# 模型下载
cache_dir = r'E:\LLM'
model_dir = snapshot_download('sentence-transformers/all-MiniLM-L6-v2', cache_dir=cache_dir)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England."
]

# initialize the SentenceTransformerEmbeddingFunction
ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name=r"E:\LLM\sentence-transformers\all-MiniLM-L6-v2",  # Specify the model name
    device="cpu"  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

queries = docs
queries_embeddings = ef(queries)
docs_embeddings = ef(docs)
print("dense dim:", ef.dim, queries_embeddings[0].shape)
print("dense dim:", ef.dim, docs_embeddings[0].shape)

# get the embeddings in specified way.
queries_embeddings = ef.encode_queries(queries)
docs_embeddings = ef.encode_documents(docs)

print("dense dim:", ef.dim, queries_embeddings[0].shape)
print("dense dim:", ef.dim, docs_embeddings[0].shape)

# 使用git下载:git clone https://hf-mirror.com/naver/splade-cocondenser-ensembledistil
splade_ef = SpladeEmbeddingFunction(model_name=r'E:\LLM\splade-cocondenser-ensembledistil')
queries = docs
queries_embeddings = splade_ef(queries)
docs_embeddings = splade_ef(docs)
print("sparse dim:", splade_ef.dim, list(queries_embeddings)[0].shape)
print("sparse dim:", splade_ef.dim, list(docs_embeddings)[0].shape)
queries_embeddings = splade_ef.encode_queries(queries)
docs_embeddings = splade_ef.encode_documents(docs)
print("sparse dim:", splade_ef.dim, list(queries_embeddings)[0].shape)
print("sparse dim:", splade_ef.dim, list(docs_embeddings)[0].shape)

# 下载模型.
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='E:\LLM')
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=r'E:\LLM\BAAI\bge-m3',  # Specify the model name
    device='cpu',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)
docs_embeddings = bge_m3_ef.encode_documents(docs)
print("Embeddings:", docs_embeddings)
print("Dense document dim:", bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape)
print("Sparse document dim:", bge_m3_ef.dim["sparse"], list(docs_embeddings["sparse"])[0].shape)
