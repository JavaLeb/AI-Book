from langgraph.store.memory import InMemoryStore
from uuid_utils import uuid7

store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid7())
memory = {"food_preference" : "I like pizza"}
store.put(namespace_for_memory, memory_id, memory)

memories = store.search(namespace_for_memory)

print('search_memories=',memories)
print(memories[-1].dict())

get_memories=store.get(namespace_for_memory, memory_id)
print('get_memories=',get_memories)

store.delete(namespace_for_memory,memory_id)

get_memories=store.get(namespace_for_memory, memory_id)
print('get_memories after delete=',get_memories)