from model_identifiers import qwen2dot5_7b_identifier
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

contact_info_schema = {
    "title": "ContactInfo",
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"}
    },
    "required": ["name", "email", "phone"]
}

agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ProviderStrategy(contact_info_schema)
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

from rich.pretty import pprint

pprint(response, expand_all=True)

print(f'structured_response={response["structured_response"]}')
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}

