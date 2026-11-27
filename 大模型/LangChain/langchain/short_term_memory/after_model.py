from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime
from model_identifiers import qwen2dot5_7b_identifier
from langchain_core.runnables import RunnableConfig


def my_print(step):
    print('+' * 50 + 'Response ' + str(step) + '+' * 50)


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""

    my_print(step=0)
    for msg in state["messages"]:
        msg.pretty_print()
    my_print(step=0)

    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]

    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}

    return None


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

response1 = agent.invoke({"messages": "hi, my name is bob"}, config)
my_print(step=1)
for message in response1["messages"]:
    message.pretty_print()
my_print(step=1)

response2 = agent.invoke({"messages": "write a password"}, config)
my_print(step=2)
for message in response2["messages"]:
    message.pretty_print()
my_print(step=2)

response3 = agent.invoke({"messages": "what's my name?"}, config)
my_print(step=3)
for message in response3["messages"]:
    message.pretty_print()
my_print(step=3)
