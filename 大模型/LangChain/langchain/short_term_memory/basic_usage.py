from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from model_identifiers import qwen2dot5_7b_identifier


def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    thread_config,
)

print(response["messages"][-1].content)  # "Hi Bob! Nice to see you here. How are you doing?"

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    thread_config,
)
print("=="*60)
print(response["messages"][-1].content)  # "You are Bob!"
