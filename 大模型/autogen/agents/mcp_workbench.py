from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from model_client import model_client
import asyncio

# 需要在PS中执行如下命令安装mcp-server-fetch:
# pip install mcp-server-fetch
# 使用StdioServerParams(command="python", args=["-m", "mcp_server_fetch"])

# 如果是uv安装的mcp-server-fetch
# 使用StdioServerParams(command="uvx", args=["mcp-server-fetch"])

# Get the fetch tool from mcp-server-fetch.
# fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])

fetch_mcp_server = StdioServerParams(command="python", args=["-m", "mcp_server_fetch"])



async def main():
    # Create an MCP workbench which provides a session to the mcp server.
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore

        fetch_agent = AssistantAgent(
            name="fetcher", model_client=model_client, workbench=workbench, reflect_on_tool_use=True
        )

        # Let the agent fetch the content of a URL and summarize it.
        result = await fetch_agent.run(task="总结如下网页中的内容： https://www.hfut.edu.cn/xxgk/gdjj.htm")
        assert isinstance(result.messages[-1], TextMessage)
        print(result.messages[-1].content)

        # Close the connection to the model client.
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
