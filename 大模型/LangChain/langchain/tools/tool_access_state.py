from langchain.messages import HumanMessage,ToolMessage
from langchain.agents import create_agent,AgentState
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.types import Command


class UserPreferencesState(AgentState):
    user_preferences: dict[str, str]


# Access custom state fields
@tool
def get_user_preference(
        pref_name: str,
        runtime: ToolRuntime
) -> str:
    """Get a user preference value."""

    preferences = runtime.state.get("user_preferences", {})
    print('正在调用get_user_preference工具，user_preferences=', preferences)

    return preferences.get(pref_name, "Not set")


@tool
def set_user_preference(user_preferences: dict[str, str], runtime: ToolRuntime[None, UserPreferencesState]) -> Command:
    """Set the user's name in the conversation state."""

    print('正在调用set_user_preference工具，user_preferences=', user_preferences)

    command=Command(
        update={
            "user_preferences": user_preferences,
            "messages": [
                ToolMessage(
                    content=f"user_preferences set to {user_preferences}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

    return command


model = init_chat_model(model="ollama:qwen2.5:7b")

agent = create_agent(
    model,
    tools=[ get_user_preference, set_user_preference],
    state_schema=UserPreferencesState
)

# First session: save user info
response = agent.invoke(
    input={"messages": [HumanMessage(content="我现在的字体大小偏好是“23px”。先查询之前我的字体大小偏好，"
                                             "然后更新我的字体大小偏好，最后再次查询我的字体大小偏好。")],
           "user_preferences": {
               "theme": "dark",
               "font_size": "18px",
               "language": "zh-Hans"}
           })
print("=== 模型回答 ===")
print(response["messages"][-1].content)

print("=== 完整运行过程 ===")
for msg in response["messages"]:
    print(f"[{msg.type}] {msg.content}")
