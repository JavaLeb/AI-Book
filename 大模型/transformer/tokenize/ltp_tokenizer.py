from ltp import LTP

ltp = LTP(path="LTP/small")
text = "今天天气真好，我们一起去公园散步吧！"
seg, hidden = ltp.seg([text])
print("分词结果:", seg[0])
