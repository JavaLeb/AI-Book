from models import qwen2dot5_7b_model as model

response = model.invoke("Provide details about the movie Inception")
print(f'type(response)={type(response)}')

print(f'response={response}')

from  rich.pretty import pprint

pprint(response)
