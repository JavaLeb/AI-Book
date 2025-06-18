import thulac

thu = thulac.thulac(seg_only=True)  # 只分词，不进行词性标注
text = "我爱自然语言处理。"
seg_list = thu.cut(text, text=True)
print(seg_list)  # 输出: 我 爱 自然 语言 处理 。
