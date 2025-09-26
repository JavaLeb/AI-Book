from io import BytesIO

import PIL
import requests
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image

import asyncio
from autogen_agentchat.agents import AssistantAgent

# Create a multi-modal message with random image and text.
pil_image = PIL.Image.open(BytesIO(requests.get("https://picsum.photos/300/200").content))
img = Image(pil_image)
multi_modal_message = MultiModalMessage(content=["Can you describe the content of this image?", img], source="user")

from model_client import multi_model_client

agent = AssistantAgent(
    name="assistant",
    model_client=multi_model_client,
)

result = asyncio.run(agent.run(task=multi_modal_message))
print(result.messages[-1].content)  # type: ignore

pil_image.show()
