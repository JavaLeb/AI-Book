from pymilvus.model.reranker import BGERerankFunction, CrossEncoderRerankFunction
from modelscope import snapshot_download

query = "What event in 1956 marked the official birth of artificial intelligence as a discipline?"
# 假设documents是第一阶段检索出的结果.
documents = [
    "In 1950, Alan Turing published his seminal paper, 'Computing Machinery and Intelligence,' proposing the Turing Test as a criterion of intelligence, a foundational concept in the philosophy and development of artificial intelligence.",
    "The Dartmouth Conference in 1956 is considered the birthplace of artificial intelligence as a field; here, John McCarthy and others coined the term 'artificial intelligence' and laid out its basic goals.",
    "In 1951, British mathematician and computer scientist Alan Turing also developed the first program designed to play chess, demonstrating an early example of AI in game strategy.",
    "The invention of the Logic Theorist by Allen Newell, Herbert A. Simon, and Cliff Shaw in 1955 marked the creation of the first true AI program, which was capable of solving logic problems, akin to proving mathematical theorems."
]

# 模型下载
cache_dir = r'E:\LLM'
model_dir = snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir=cache_dir)

bge_rf = BGERerankFunction(model_name=r"E:\LLM\BAAI\bge-reranker-v2-m3", device="cuda:0")
results = bge_rf(query, documents, top_k=3)
print('bge_rf:', len(results), results)

# 使用git下载: git clone https://hf-mirror.com/cross-encoder/ms-marco-MiniLM-L6-v2
ce_rf = CrossEncoderRerankFunction("E:\LLM\ms-marco-MiniLM-L6-v2", device="cpu")
results = ce_rf(query, documents, top_k=3)
print('ce_rf:', len(results), results)
