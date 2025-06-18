import jieba

# 精确模式
text = "我爱自然语言处理。"
seg_list = jieba.lcut(text, cut_all=False)
print("精确模式:", seg_list)  # 输出: ['我', '爱', '自然语言', '处理', '。']

# 全模式
seg_list = jieba.lcut(text, cut_all=True)
print("全模式:", seg_list)  # 输出: ['我', '爱', '自然', '自然语言', '语言', '处理', '。']

# 搜索引擎模式
seg_list = jieba.lcut_for_search(text)
print("搜索引擎模式:", seg_list)  # 输出: ['我', '爱', '自然', '语言', '自然语言', '处理', '。']
