import spacy

# 加载英文模型
# 手工下载地址: https://github.com/explosion/spacy-models/releases
nlp = spacy.load("en_core_web_sm")

text = "Natural Language Processing (NLP) is an important field of AI."

# 分词处理
doc = nlp(text)
tokens = [token.text for token in doc]

print("spaCy分词:", tokens)
# 输出: ['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'is', 'an', 'important', 'field', 'of', 'AI', '.']