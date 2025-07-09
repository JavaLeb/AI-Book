from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
text = "PyTorch is awesome!"
tokens = tokenizer(text)
print(tokens)
# 输出: ['pytorch', 'is', 'awesome', '!']