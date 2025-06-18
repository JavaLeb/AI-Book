from modelscope import AutoTokenizer

# 加载基础BERT中文模型
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese",
                                          clean_up_tokenization_spaces=True)

# 获取子词级分词
text = "我爱自然语言处理。"
tokens = tokenizer.tokenize(text)
# 输出：['我', '爱', '自', '然', '语', '言', '处', '理', '。']
print(tokens)
