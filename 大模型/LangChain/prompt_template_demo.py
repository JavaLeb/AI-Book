from langchain_core.prompts import ChatPromptTemplate
from ollama_model import model

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
prompt_value = prompt_template.invoke({"topic": "cats"})
print('prompt_value=', prompt_value)

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])
chat_prompt_value = prompt_template.invoke({"topic": "cats"})
print('chat_prompt_value=', chat_prompt_value)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

# Simple example with one message
prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})

# More complex example with conversation history
messages_to_pass = [
    HumanMessage(content="What's the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="And what about Germany?")
]

formatted_prompt = prompt_template.invoke({"msgs": messages_to_pass})
print(formatted_prompt)

# AIMessage.
from langchain_core.messages import HumanMessage

ai_message = model.invoke([HumanMessage("Tell me a joke")])
print(type(ai_message))
print(ai_message)

for chunk in model.stream([HumanMessage("what color is the sky?")]):
    print('chunk=', chunk)

# system_template = "Translate the following text from English into {language}"  # 系统模版.
# user_template = "text:{text}"  # 用户模版.
# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", user_template)]
# )
# print('prompt_template=', prompt_template)
#
# prompt = prompt_template.invoke({"language": "中文", "text": "hi!"})
# print('prompt', prompt)
#
# messages = prompt.to_messages()
# print('messages=', messages)
#
# response = model.invoke(prompt)
# print('content=', response.content)
