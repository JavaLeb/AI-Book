from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from model_identifiers import llama3dot2_3b_identifier

def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


DB_URI = "postgresql://postgres:root@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # auto create tables in PostgreSQL
    agent = create_agent(
        model=llama3dot2_3b_identifier,
        tools=[get_user_info],
        checkpointer=checkpointer,
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
    print("==" * 60)
    print(response["messages"][-1].content)  # "You are Bob!"
