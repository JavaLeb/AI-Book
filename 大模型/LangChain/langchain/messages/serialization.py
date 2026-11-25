from langchain.messages import HumanMessage
from langchain_core.load import dumpd, load

message = HumanMessage("What is the capital of France?")

# Serialize to a plain dict
serialized = dumpd(message)
print(f'type:{type(serialized)},serialized: {serialized}')

# Deserialize back to a message object
restored = load(serialized)
print(f'type:{type(restored)},restored: {restored}')