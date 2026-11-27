from model_identifiers import qwen2dot5_7b_identifier
from dataclasses import dataclass
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


@dataclass
class ProductReview:
    """Analysis of a product review."""
    rating: int | None  # The rating of the product (1-5)
    sentiment: Literal["positive", "negative"]  # The sentiment of the review
    key_points: list[str]  # The key points of the review


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(ProductReview)
)

response = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

from rich.pretty import pprint

pprint(response, expand_all=True)

print(f'structured_response={response["structured_response"]}')
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
