from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ollama_model import model
import asyncio  # 确保导入了 asyncio


prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

# ✅ 将 async for 包在一个 async 函数里
async def run_stream():
    async for event in chain.astream_events({"topic": "parrot"}):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(event, end="|", flush=True)

# ✅ 使用 asyncio.run 来运行这个异步函数
if __name__ == "__main__":
    asyncio.run(run_stream())