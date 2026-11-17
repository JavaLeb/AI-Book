from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain.models.models import qwen2dot5_7b_instruct_model as model
import asyncio

async def main():
    async with MCPAdapter("https://example.com/mcp") as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
        # 注意：在 async 函数里调用 agent，用 ainvoke（异步版）
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "今天有什么好消息？"}]}
        )
        print(result["messages"][-1].content)

asyncio.run(main())