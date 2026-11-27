from rich.pretty import pprint

from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from model_identifiers import qwen2dot5_7b_identifier


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")


class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails],
                                 handle_errors=False),  # Default: handle_errors=True

)

response = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Extract ALL information using EVERY available tool. "
                             "You MUST call both ContactInfo and EventDetails. "
                             "Info: John Doe (john@email.com) is organizing Tech Conference on March 15th."
                  }]
})

# pprint(response, expand_all=True)

for message in response["messages"]:
    message.pretty_print()
