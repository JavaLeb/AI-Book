from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from models import qwen2dot5_7b_model as model


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""

    print("调用get_user_info，查询user_id=", user_id)
    store = runtime.store
    user_info = store.get(("users",), user_id)

    return str(user_info.value) if user_info else "Unknown user"


# Update memory
@tool
def save_user_info(user_id: str, name: str, age: int, email: str, runtime: ToolRuntime) -> str:
    """Save user info."""
    print("调用save_user_info，保存user_id=", user_id)
    store = runtime.store
    store.put(("users",), user_id, {"name": name, "age": age, "email": email})

    return "Successfully saved user info."


store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
save_resp = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})
print("save_resp=", save_resp)
print("save-message=", save_resp["messages"][-1].content)

# Second session: get user info
get_resp = agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
print("get_resp=", get_resp)
print("get-message=", get_resp["messages"][-1].content)
