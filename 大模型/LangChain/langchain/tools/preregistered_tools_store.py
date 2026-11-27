from langchain_core.tools import tool

from models import qwen2dot5_7b_model as model
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore


@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    return f"[search] results for: {query}"


@tool
def analysis_tool(data: str) -> str:
    """Analyze data and produce insights."""
    return f"[analysis] insights for: {data}"


@tool
def export_tool(fmt: str) -> str:
    """Export results to a file format."""
    return f"[export] exported as {fmt}"


@dataclass
class Context:
    user_id: str


@wrap_model_call
def store_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    user_id = request.runtime.context.user_id

    # Read from Store: get user's enabled features
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        # Only include tools that are enabled for this user
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)
        reason = f"user={user_id}, enabled={enabled_features}"
    else:
        reason = f"user={user_id}, 无配置 -> 不过滤（暴露全部）"

    exposed = [t.name for t in request.tools]
    print(f"[store_based_tools] {reason} -> 暴露工具: {exposed}")

    return handler(request)


store = InMemoryStore()

store.put(
    ("features",),
    "alice",
    {"enabled_tools": ["search_tool", "analysis_tool"]},
)
store.put(
    ("features",),
    "bob",
    {"enabled_tools": ["search_tool", "analysis_tool", "export_tool"]},
)

agent = create_agent(
    model=model,
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=store,
    system_prompt="你是数据助手。根据用户需求调用合适的工具。"
)

for user_id in ["alice", "bob", "carol"]:
    print("\n" + "=" * 25 + f" user={user_id} " + "=" * 25)
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "帮我搜索 LangChain 的资料，分析一下，然后导出结果。",
                }
            ]
        },
        context=Context(user_id=user_id),
    )
