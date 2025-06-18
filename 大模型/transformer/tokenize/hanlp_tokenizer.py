from pyhanlp import HanLP

# 运行时会下载https://file.hankcs.com/hanlp/data-for-*.zip
text = "我爱自然语言处理。"
words = HanLP.segment(text)
print(words)  # 输出: [我/rr, 爱/v, 自然语言处理/nz, 。/w]

import hanlp

# 加载预训练中文分词模型
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tokens = tokenizer(text)
print(tokens)  # 输出: ['自然语言', '处理', '是', '人工智能', '的', '重要', '领域']
