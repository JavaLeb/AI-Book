from langgraph.store.memory import InMemoryStore
from uuid_utils import uuid7

store = InMemoryStore()

user = "alice"
store.put((user, "memories",'GB'), str(uuid7()), {"my_mem": "1GB"})
store.put((user, "preferences",'food'), str(uuid7()), {"food_preference": "I like pizza"})


namespaces = store.list_namespaces(prefix=("alice",), max_depth=2)
print('namespaces:', namespaces)

# Return up to 100 items stored under ("alice", "memories").
items = store.search(("alice", "memories"), limit=100)
print('items:', items)

page_size = 1
offset = 0
current_page=0
while True:
    page = store.search(("alice",), limit=page_size, offset=offset)
    if not page:
        break
    for item in page:
        print('current_page=',current_page,'item=',item)
    offset += page_size
    current_page += 1