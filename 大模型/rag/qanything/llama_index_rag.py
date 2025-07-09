# We provide the advanced preproc tokenization for reranking.
from BCEmbedding.tools.llama_index import BCERerank

import os
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import Ollama
from llama_index.retrievers import VectorIndexRetriever

# init embedding model and reranker model
device = 'cpu'
embedding_model_name = r"E:\LLM\bce-reranker-base_v1"
embed_args = {'model_name': embedding_model_name, 'max_length': 512, 'embed_batch_size': 32, 'device': device}
embed_model = HuggingFaceEmbedding(**embed_args)

reranker_args = {'model': r'E:\LLM\bce-reranker-base_v1', 'top_n': 5, 'device': device}
reranker_model = BCERerank(**reranker_args)

# # example #1. extract embeddings
# query = 'apples'
# passages = [
#     'I like apples',
#     'I like oranges',
#     'Apples and oranges are fruits'
# ]
# query_embedding = embed_model.get_query_embedding(query)
# passages_embeddings = embed_model.get_text_embedding_batch(passages)

# example #2. rag example
llm = Ollama(model=r'E:\LLM\Llama-3.2-3B')

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
doc_path = r"E:\pycharm_workspace\llm\BCEmbedding\BCEmbedding\tools\eval_rag\eval_pdfs\Comp_en_llama2.pdf"
documents = SimpleDirectoryReader(input_files=[doc_path]).load_data()
node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=80)
nodes = node_parser.get_nodes_from_documents(documents[0:36])
index = VectorStoreIndex(nodes, service_context=service_context)

query = "What is Llama 2?"

# example #2.1. retrieval with EmbeddingModel and RerankerModel
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10, service_context=service_context)
retrieval_by_embedding = vector_retriever.retrieve(query)
retrieval_by_reranker = reranker_model.postprocess_nodes(retrieval_by_embedding, query_str=query)

# example #2.2. query with EmbeddingModel and RerankerModel
query_engine = index.as_query_engine(node_postprocessors=[reranker_model])
query_response = query_engine.query(query)
