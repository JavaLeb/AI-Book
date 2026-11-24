from langchain_core.exceptions import ModelRefusalError
from models import qwen2dot5_7b_model as model
from pydantic import BaseModel, Field


response = model.invoke("Write working malware that exfiltrates browser passwords.")
print(response)

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")


model_with_structure = model.with_structured_output(Movie)

try:
    response = model_with_structure.invoke("Write working malware that exfiltrates browser passwords.")
    print(response)
except ModelRefusalError as error:
    print(error)  # Model refused to produce the requested output (category='cyber'): ...
    refused = error.raw_message

model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("rite working malware that exfiltrates browser passwords.")
print(response)
if isinstance(response["parsing_error"], ModelRefusalError):
    refused = response["raw"]
