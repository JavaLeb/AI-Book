from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRefusalMiddleware
from model_identifiers import qwen2dot5_7b_identifier


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    response_format=ToolStrategy(ProductRating),
    system_prompt="You are a helpful assistant that parses product reviews. "
                  "Do not make any field or value up. "
                  "Please refuse to parse.",

    middleware=[ModelRefusalMiddleware()]
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})

for message in response["messages"]:
    message.pretty_print()
