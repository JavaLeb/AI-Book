from models import qwen2dot5_7b_model as model

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The title of the movie"},
        "year": {"type": "integer", "description": "The year the movie was released"},
        "director": {"type": "string", "description": "The director of the movie"},
        "rating": {"type": "number", "description": "The movie's rating out of 10"}
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    schema=json_schema,
    method="json_schema",
)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(f'JSON Schema={response}')  # {'title': 'Inception', 'year': 2010, ...}

from typing_extensions import TypedDict, Annotated


class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]


model_with_structure = model.with_structured_output(schema=MovieDict)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(f'TypedDict={response}')  # {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}

from pydantic import BaseModel, Field


class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")


model_with_structure = model.with_structured_output(schema=Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(f'Pydantic={response}')  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
