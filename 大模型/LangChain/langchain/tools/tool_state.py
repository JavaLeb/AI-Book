from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from models import qwen2dot5_7b_model as model


# 1. 自定义状态：继承 AgentState，增加业务字段
class CustomState(AgentState):
    user_name: str
    user_preferences: dict[str, str]


# 2. 读取 messages 状态
@tool
def get_last_user_message(runtime: ToolRuntime[None, CustomState]) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state["messages"]

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    return "No user messages found"


# 3. 读取自定义状态字段 user_preferences
@tool
def get_user_preference(
        pref_name: str,
        runtime: ToolRuntime[None, CustomState],
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")


# 4. 读取自定义状态字段 user_name，用于验证更新是否生效
@tool
def get_user_name(runtime: ToolRuntime[None, CustomState]) -> str:
    """Get the user's name from conversation state."""
    return runtime.state.get("user_name", "Not set")


# 5. 更新 state：返回 Command，并附带 ToolMessage
@tool
def set_user_name(
        new_name: str,
        runtime: ToolRuntime[None, CustomState],
) -> Command:
    """Set the user's name in the conversation state."""
    return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(
                    content=f"User name set to {new_name}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# 6. 创建模型和 agent
checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    tools=[
        get_last_user_message,
        get_user_preference,
        get_user_name,
        set_user_name,
    ],
    state_schema=CustomState,
    checkpointer=checkpointer,
    system_prompt=(
        "你是一个助手。"
        "当用户要求记住名字时，调用 set_user_name。"
        "当用户询问名字时，调用 get_user_name。"
        "当用户询问偏好时，调用 get_user_preference。"
        "当用户询问最后一条用户消息时，调用 get_last_user_message。"
    ),
)

# 同一个 thread_id 表示同一段短期会话记忆
config = {"configurable": {"thread_id": "thread-1"}}

# 第一轮：更新 user_name，同时初始化 user_preferences
result1 = agent.invoke(
    {
        "messages": [HumanMessage("请把我的名字设置为 Alice。")],
        "user_preferences": {"language": "中文", "theme": "dark"},
    },
    config=config,
)

print("第一轮最终回复：", result1["messages"][-1].content)
print("第一轮 user_name：", result1.get("user_name"))
print("第一轮 user_preferences：", result1.get("user_preferences"))

# 第二轮：读取上一轮更新后的状态
result2 = agent.invoke(
    {
        "messages": [HumanMessage("请读取当前用户名、language 偏好，并告诉我最后一条用户消息。")]
    },
    config=config,
)

print("第二轮最终回复：", result2["messages"][-1].content)
print("第二轮 user_name：", result2.get("user_name"))
print("第二轮 user_preferences：", result2.get("user_preferences"))
