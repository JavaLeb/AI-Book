from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from model_identifiers import llama3dot2_3b_identifier


def my_print(step):
    print('+' * 50 + 'Response ' + str(step) + '+' * 50)


def delete_all_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


agent = create_agent(
    model=llama3dot2_3b_identifier,
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
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
