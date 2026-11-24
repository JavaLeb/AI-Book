from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from models import qwen2dot5_7b_instruct_model as basic_model
from models import qwen2dot5_7b_model as advanced_model


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 1:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,  # Default model
    tools=[],
    middleware=[dynamic_model_selection]
)

messages = [{"role": "user", "content": "hello"}]

response = agent.invoke(input={"messages": messages})
print(response['messages'][-1].response_metadata)

messages.append({"role": "user", "content": "my name is Tom!"})
response = agent.invoke(input={"messages": messages})
print(response['messages'][-1].response_metadata)
