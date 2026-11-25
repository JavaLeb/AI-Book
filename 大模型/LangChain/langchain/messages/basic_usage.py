from models import qwen2dot5_7b_model as model

response = model.invoke(input="Write a haiku about spring")
print(f'Text prompts Response: {response}')

from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(input=messages)
print(f'Message prompts Response: {response}')

messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(input=messages)
print(f'Dictionary format Response: {response}')