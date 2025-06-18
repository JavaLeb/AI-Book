import nltk

# 如果代码无法下载, 使用git手工下载. 下载命令如下:
# git clone https://github.com/nltk/nltk_data
# 将nltk_data\packages\tokenizers目录下的punkt.zip和punkt_tab.zip解压,
# 将tokenizers文件夹放至报错日志Searched in显示的任一目录下即可,例如: D:\SOFT\Python312\Lib\nltk_data.
nltk.download('punkt')
nltk.download('punkt_tab')
print(nltk.data.path)  # 输出所有可能的搜索路径列表

# 测试分词
from nltk.tokenize import word_tokenize

text = "I love natural language processing."
tokens = word_tokenize(text)
print("分词结果:", tokens)  # ['I', 'love', 'natural', 'language', 'processing', '.']
