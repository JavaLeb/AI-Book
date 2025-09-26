from autogen_agentchat.messages import TextMessage

# source: The name of the agent that sent this message.
# content: The content of this message.
text_message = TextMessage(content="Hello, world!", source="User")
print(text_message)

from io import BytesIO

import requests
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from PIL import Image

pil_image = Image.open(BytesIO(requests.get("https://picsum.photos/300/200").content))
img = AGImage(pil_image)
text = "Can you describe the content of this image?"
multi_modal_message = MultiModalMessage(content=[text, img], source="User")
print(multi_modal_message)
