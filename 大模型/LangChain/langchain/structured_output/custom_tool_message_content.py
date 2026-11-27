from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from model_identifiers import qwen2dot5_7b_identifier
from rich.pretty import pprint


class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(schema=MeetingAction)
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})

pprint(response, expand_all=True)

print('='*60)

tool_message_agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
    )
)

response = tool_message_agent.invoke({
    "messages": [
        {"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})

pprint(response, expand_all=True)
