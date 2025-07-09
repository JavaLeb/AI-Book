from BCEmbedding.tools.langchain import BCERerank

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever

# init embedding model.
embedding_model_name = r"E:\LLM\bce-embedding-base_v1"
encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': True}
embed_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    encode_kwargs=encode_kwargs
)

# init reranker model.
reranker_model_name = r"E:\LLM\bce-reranker-base_v1"
reranker_args = {'model': reranker_model_name, 'top_n': 5}
reranker_model = BCERerank(**reranker_args)

# init documents.
doc_path = r"E:\pycharm_workspace\llm\BCEmbedding\BCEmbedding\tools\eval_rag\eval_pdfs\Comp_en_llama2.pdf"
documents = PyPDFLoader(doc_path).load()  # 每一页是一个document.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)  # 文本切分.

# example 1. retrieval with embedding and reranker
# 使用FAISS向量数据库进行初步检索.
# texts: 待检索的文档集合（如文本分块列表）.
# embed_model: 嵌入模型（如OpenAI的text-embedding-ada-002），用于将文本转换为向量.
# distance_strategy: 相似度计算方式（适用于余弦相似度）.
# search_type: 基于向量相似度搜索.
# k=10: 返回前10个最相关文档.
# score_threshold=0.3：仅返回相似度分数≥0.3的文档.
retriever = FAISS.from_documents(
    texts, embed_model,
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
    search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10})

# 对初步检索结果进行优化.
# base_retriever: 第一阶段的基础检索器（FAISS）.
# base_compressor: 重排序模型.
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_model, base_retriever=retriever
)

response = compression_retriever.get_relevant_documents("What is Llama 2?")
print(response)
