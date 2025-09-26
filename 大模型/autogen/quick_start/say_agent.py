import asyncio
from autogen_agentchat.agents import AssistantAgent
from model_client import model_client


async def main() -> None:
    agent = AssistantAgent("assistant", model_client=model_client)
    print(await agent.run(task="Say 'Hello World!'"))
    await model_client.close()


asyncio.run(main())
