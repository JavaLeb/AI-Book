from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from typing import Union
from pydantic import BaseModel, Field
from model_identifiers import qwen2dot5_7b_identifier


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")


class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")


def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=custom_error_handler
    )  # Default: handle_errors=True
)

response = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Extract ALL information using EVERY available tool. "
                             "You MUST call both ContactInfo and EventDetails. "
                             "Info: John Doe (john@email.com) is organizing Tech Conference on March 15th."}]
})

for message in response["messages"]:
    message.pretty_print()
