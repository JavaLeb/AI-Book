from snownlp import SnowNLP

text = "我爱自然语言处理。"
s = SnowNLP(text)
tokens = s.words
print(tokens)  # 输出: ['我', '爱', '自然', '语言', '处理', '。']
