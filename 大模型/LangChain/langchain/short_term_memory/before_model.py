from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any
from model_identifiers import qwen2dot5_7b_identifier


def my_print(step):
    print('+' * 50 + 'Response ' + str(step) + '+' * 50)


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    # 如果消息长度大于3，裁剪.
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

response1 = agent.invoke({"messages": "hi, my name is bob"}, config)
my_print(step=1)
for message in response1["messages"]:
    message.pretty_print()
my_print(step=1)

response2 = agent.invoke({"messages": "write a short poem about cats"}, config)
my_print(step=2)
for message in response2["messages"]:
    message.pretty_print()
my_print(step=2)

response3 = agent.invoke({"messages": "now do the same but for dogs"}, config)
my_print(step=3)
for message in response3["messages"]:
    message.pretty_print()
my_print(step=3)

final_response = agent.invoke({"messages": "what's my name?"}, config)
my_print(step=4)
for message in final_response["messages"]:
    message.pretty_print()
my_print(step=4)
