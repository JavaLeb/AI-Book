from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.agents import AgentState
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from models import qwen2dot5_7b_model as model


@tool
def public_search(query: str) -> str:
    """Search public information, available to everyone."""
    return f"[public] results for: {query}"


@tool
def private_search(query: str) -> str:
    """Search private/user-specific information, requires authentication."""
    return f"[private] results for: {query}"


@tool
def advanced_search(query: str) -> str:
    """Advanced search with deeper analysis, available later in conversation."""
    return f"[advanced] results for: {query}"


class AuthState(AgentState):
    authenticated: bool


@wrap_model_call
def state_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    # Read from State: check if user has authenticated
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # Only enable sensitive tools after authentication
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
        reason = '未认证'
    elif message_count < 5:
        # Limit tools early in conversation
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)
        reason = f"已认证, messages={message_count} < 5"
    else:
        reason = f"已认证, messages={message_count} >= 5"

    exposed = [t.name for t in request.tools]
    print(f"[state_based_tools][middleware] {reason} -> 暴露工具: {exposed}")

    return handler(request)


agent = create_agent(
    model=model,
    tools=[public_search, private_search, advanced_search],
    state_schema=AuthState,
    middleware=[state_based_tools],
    checkpointer=InMemorySaver(),
    system_prompt="你是搜索助手。根据用户问题调用合适的搜索工具。",
)

print("=" * 30 + "场景一：未认证，期望只暴露 public_search" + "=" * 30)
config = {"configurable": {"thread_id": "thread-unauthenticated"}}
agent.invoke(
    input={"messages": [{"role": "user", "content": "搜索一下 LangChain 是什么"}]},
    config=config,
)


config2 = {"configurable": {"thread_id": "thread-authenticated"}}
print("=" * 30 +"场景二：已认证但消息数 < 5，期望排除 advanced_search"+ "=" * 30)
agent.invoke(
    {
        "authenticated": True,
        "messages": [{"role": "user", "content": "搜索我的账户相关信息"}],
    },
    config=config2,
)


print("=" * 30 +"场景三：同一 thread 继续多轮，期望消息数 >= 5 后暴露全部工具"+ "=" * 30)
for i in range(1, 4):
    print(f"\n--- 第 {i} 轮追加对话 ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": f"继续搜索第 {i} 个问题"}]},
        config=config2,
    )
