from langchain_text_splitters import CharacterTextSplitter, \
    TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader

text = """LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
它支持文档加载、文本分割、嵌入和检索等能力。
你可以用它来构建问答系统、聊天机器人和 RAG 应用。"""

# separator优先级高于chunk_size.
text_splitter = CharacterTextSplitter(
    chunk_size=3, chunk_overlap=0, separator='')
texts = text_splitter.split_text(text)
print('texts=', texts)

# 输出有乱码.
text_splitter = TokenTextSplitter(
    chunk_size=3, chunk_overlap=0)
texts = text_splitter.split_text(text)
print('texts=', texts)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_splitter.split_text(text)
print('texts=', texts)