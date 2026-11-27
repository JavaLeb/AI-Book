from model_identifiers import qwen2dot5_7b_identifier
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

product_review_schema = {
    "title": "ProductReview",
    "type": "object",
    "description": "Analysis of a product review.",
    "properties": {
        "rating": {
            "type": ["integer", "null"],
            "description": "The rating of the product (1-5)",
            "minimum": 1,
            "maximum": 5
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"],
            "description": "The sentiment of the review"
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The key points of the review"
        }
    },
    "required": ["sentiment", "key_points"]
}

agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[],
    response_format=ToolStrategy(product_review_schema)
)

response = agent.invoke({
    "messages": [{"role": "user",
                  "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

from rich.pretty import pprint

pprint(response, expand_all=True)

print(f'structured_response={response["structured_response"]}')
# {'sentiment': 'positive', 'rating': 5, 'key_points': ['Great product', 'Fast shipping', 'Expensive']}
