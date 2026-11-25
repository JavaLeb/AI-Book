import asyncio

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from models import qwen2dot5_7b_model as model

async def main():
    async with MCPAdapter("http://localhost:8000/mcp") as adapter:
        tools = await adapter.list_tools()
        for tool in tools:
            print('tool:', tool)

        agent =create_agent(model=model,tools=tools)
        response = await agent.ainvoke({"messages":[{"role":"user","content":"计算5+2等于多少？并查询合肥的天气。"}]})
        print(response['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())