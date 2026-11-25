from collections.abc import Callable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from models import qwen2dot5_7b_model as model


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. b must not be zero."""
    if b == 0:
        # 故意抛异常，模拟工具内部失败
        raise ValueError("division by zero is not allowed")
    return a / b


@wrap_tool_call
def handle_tool_errors(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Convert tool exceptions into ToolMessages the model can handle."""
    tool_name = request.tool_call["name"]
    try:
        return handler(request)
    except Exception as e:
        print(f"[middleware] 捕获到工具 {tool_name} 的异常: {type(e).__name__}: {e}")
        return ToolMessage(
            content=(
                f"Tool '{tool_name}' failed: {type(e).__name__}: {e}. "
                "Please check the input and try again."
            ),
            tool_call_id=request.tool_call["id"],
        )


agent = create_agent(
    model=model,
    tools=[divide],
    middleware=[handle_tool_errors],
    system_prompt=(
        "你是一个计算助手。用户要求做除法时调用 divide 工具，"
        "要求做加法时调用 add 工具。如果工具返回错误，请向用户解释原因。"
    ),
)

# ---------- 场景一：工具内部抛异常，被 middleware 兜住 ----------
print("=" * 30 + "场景一：divide(10, 0) 会抛 ValueError" + "=" * 30)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "请计算 10 除以 0。"}]}
)

for message in response["messages"]:
    message.pretty_print()

# ---------- 场景二：正常调用，对比结果 ----------
print("=" * 30 + "场景二：divide(10, 2) 正常返回" + "=" * 30)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "请计算 10 除以 2。"}]}
)

for message in response["messages"]:
    message.pretty_print()
