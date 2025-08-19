from langchain_core.tools import tool
from ollama_model import model, chat_ollama_model


# Tool Creation.
# 函数名add, 类型（参数类型，返回值类型）, 文档字符串, 函数体 都应该作为tool schema的一部分.
# 定义一个好的，具有描述性的schema是提示工程的扩展，可以让模型表现的更好.
@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


@tool
def minus(a: int, b: int) -> int:
    """Minus a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b


# @tool注解的函数可以调用invoke.
result = add.invoke({"a": 2, "b": 3})
print('function invoke result=', result)

print('函数名:', add.name)  # multiply
print('函数描述:', add.description)  # Multiply two numbers.
print('函数参数:', add.args)

# Tool Binding.
tools = [minus, multiply, add]
llm_with_tools = model.bind_tools(tools)

# 未绑定工具, 模型输出.
print(model.invoke("What is 2 multiplied by 3?"))

# Tool Calling, 模型输出.
messages = ["What is 2 added by 3?", "What is 2 multiplied by 3?"]
ai_msg = llm_with_tools.invoke(messages)
print(type(ai_msg))
print('ai_msg=', ai_msg)
print(ai_msg.tool_calls)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply, "minus": minus}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    print(type(tool_msg), 'tool_msg=', tool_msg)
    messages.append(tool_msg)

result = llm_with_tools.invoke(messages)

print('result=', result)
