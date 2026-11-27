from model_identifiers import qwen2dot5_7b_identifier
from typing_extensions import TypedDict
from langchain.agents import create_agent


class ContactInfo(TypedDict):
    """Contact information for a person."""
    name: str  # The name of the person
    email: str  # The email address of the person
    phone: str  # The phone number of the person


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

from rich.pretty import pprint

pprint(response, expand_all=True)

print(f'structured_response={response["structured_response"]}')
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
