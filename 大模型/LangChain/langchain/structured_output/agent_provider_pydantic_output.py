from pydantic import BaseModel, Field
from langchain.agents import create_agent
from model_identifiers import qwen2dot5_7b_instruct_identifier


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


agent = create_agent(
    model=qwen2dot5_7b_instruct_identifier,
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

from rich.pretty import pprint

pprint(response, expand_all=True)

print(f'structured_response={response["structured_response"]}')
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
