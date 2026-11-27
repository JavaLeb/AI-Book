from model_identifiers import qwen2dot5_7b_identifier
from langchain.agents import create_agent

agent = create_agent(model=qwen2dot5_7b_identifier,
                     system_prompt="You are a helpful assistant")

response = agent.invoke(input={"message": [{"role": "user", "content": "Hello"}]})

print(f'type(response)={type(response)}')
print(f'response={response}')

from rich.pretty import pprint
pprint(response, expand_all=True)
