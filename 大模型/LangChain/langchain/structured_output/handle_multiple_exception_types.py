from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from model_identifiers import qwen2dot5_7b_identifier


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(schema=ProductRating,
                                 handle_errors=(ValueError, TypeError)  # Retry on ValueError and TypeError
                                 ),  # Default: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})

for message in response["messages"]:
    message.pretty_print()
