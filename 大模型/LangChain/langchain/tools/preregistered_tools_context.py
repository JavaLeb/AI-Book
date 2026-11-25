from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

from langchain_core.tools import tool

from models import qwen2dot5_7b_model as model


@tool
def read_data() -> str:
    """Read data"""
    print("Read data")

    return 'Read data'


@tool
def write_data() -> str:
    """Write data"""
    print("Write data")

    return 'Write data'


@tool
def delete_data() -> str:
    """Delete data"""
    print("Delete data")

    return 'Delete data'


@dataclass
class Context:
    user_role: str


@wrap_model_call
def context_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    # Read from Runtime Context: get user role
    if request.runtime is None or request.runtime.context is None:
        # If no context provided, default to viewer (most restrictive)
        user_role = "viewer"
    else:
        user_role = request.runtime.context.user_role

    if user_role == "admin":
        # Admins get all tools
        pass
    elif user_role == "editor":
        # Editors can't delete
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # Viewers get read-only tools
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model=model,
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)

messages = [{"role": "user", "content": "请调用所有工具"}]

response = agent.invoke(input={"messages": messages}, context=Context(user_role="admin"))
for message in response['messages']:
    message.pretty_print()

response = agent.invoke(input={"messages": messages}, context=Context(user_role="editor"))
for message in response['messages']:
    message.pretty_print()
