import time
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from models import qwen2dot5_7b_model as model


@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    writer(f"正在查询 {city} 的天气数据...")
    time.sleep(1)  # 模拟耗时操作（如调用外部 API）

    writer(f"已获取 {city} 的天气数据，正在解析...")
    time.sleep(0.5)

    writer(f"{city} 天气查询完成")
    return f"{city} 今天晴朗，气温 25°C。"


agent = create_agent(
    model=model,
    tools=[get_weather],
)

# 使用 stream_mode="custom" 消费工具发出的进度事件
for event in agent.stream(
    {"messages": [{"role": "user", "content": "查一下合肥的天气"}]},
    stream_mode="custom",
):
    print(f"[进度] {event}")