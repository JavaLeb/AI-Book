import asyncio
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

from models import qwen2dot5_7b_model as model


async def main():
    # 公共 LangChain 文档 MCP 服务器，无需 API Key
    async with MCPAdapter("https://docs.langchain.com/mcp") as adapter:
        # 发现服务器暴露的工具，并适配为 LangChain 工具
        tools = await adapter.list_tools()

        print("发现的 MCP 工具：")
        for t in tools:
            print(f"  - {t.name}: {t.description[:80]}...")

        # 像普通工具一样交给 create_agent
        agent = create_agent(model, tools)

    # 注意：async with 退出后，tools 仍然可用（工具持有客户端）
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "LangChain agent 如何添加短期记忆（short-term memory）？",
                }
            ]
        }
    )

    print("\n最终回答：")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())