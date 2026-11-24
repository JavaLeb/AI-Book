from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

model_identifier = "ollama:qwen2.5:7b"

configurable_model = init_chat_model(temperature=0)
response = configurable_model.invoke(
    input="what's your name",
    config={"configurable": {"model": model_identifier}},
)
print(response)

first_model = init_chat_model(
    model=model_identifier,
    temperature=0,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens"),
    config_prefix="first",  # Useful when you have a chain with multiple models
)
response = first_model.invoke("what's your name")
print(response)

response = first_model.invoke(
    "what's your name",
    config={
        "configurable": {
            "first_model": model_identifier,
            "first_temperature": 0.5,
            "first_max_tokens": 100,
        }
    },
)
print(response)


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")


class GetPopulation(BaseModel):
    """Get the current population in a given location"""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")


model = init_chat_model(temperature=0)
model_with_tools = model.bind_tools([GetWeather, GetPopulation])

response = model_with_tools.invoke(
    "what's bigger in 2024 LA or NYC", config={"configurable": {"model": model_identifier}}
).tool_calls
print(response)

response = model_with_tools.invoke(
    "what's bigger in 2024 LA or NYC",
    config={"configurable": {"model": model_identifier}},
).tool_calls
print(response)
