from langchain.agents import create_agent
from langchain.tools import tool
from models import qwen2dot5_7b_model as model

@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order."""
    # In production, query your order management system here
    return f"Order {order_id} is shipped and will arrive in 2 days."


agent = create_agent(
    model=model,
    tools=[fetch_order_status],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of order #12345?"}]
})
print(result["messages"][-1].content)
# The agent returns the tool output directly without another LLM call:
# "Order 12345 is shipped and will arrive in 2 days."