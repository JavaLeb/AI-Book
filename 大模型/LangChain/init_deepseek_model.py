import getpass
import os

if not os.environ.get("DEEPSEEK_API_KEY"):
    # os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

    # 直接绕过getpass，提示输入.
    os.environ["DEEPSEEK_API_KEY"] = 'sk-9a6e037ac00b4b2190340bc820762610'
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek-chat", model_provider="deepseek")
print('模型初始化完成')
