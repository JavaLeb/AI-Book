from autogen_core.tools import FunctionTool

from tools import web_search

# This step is automatically performed inside the AssistantAgent if the tool is a Python function.
web_search_function_tool = FunctionTool(web_search, description="Find information on the web")
# The schema is provided to the model during AssistantAgent's on_messages call.
print(web_search_function_tool.schema)
