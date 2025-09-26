from autogen_agentchat.agents import AssistantAgent
import asyncio
from tools import web_search

from model_client import model_client

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)

result = asyncio.run(agent.run(task="Find information on AutoGen"))
print(result.messages)
