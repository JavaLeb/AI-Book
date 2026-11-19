from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from uuid_utils import uuid7

embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    # base_url="http://localhost:11434"   # 远程Ollama地址按需修改
)

store = InMemoryStore(
    index={
        "embed": embedding,  # Embedding provider
        # "embed": init_embeddings("ollama:nomic-embed-text"),
        "dims": 768,  # Embedding dimensions
        "fields": ["food_preference", "$"]  # Fields to embed
    }
)

user_id = "1"
namespace_for_memory = (user_id, "memories")

memories_to_add = [
    {"food_preference": "I like pizza"},
    {"food_preference": "I hate broccoli"},
    {"food_preference": "My favorite drink is lemon tea"}
]
for item in memories_to_add:
    mid = str(uuid7())
    store.put(namespace_for_memory, mid, item)
    print(f"✅ 写入 memory_id={mid}, content={item}")

store.put(
    namespace_for_memory,
    str(uuid7()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # Only embed "food_preferences" field
)

store.put(
    namespace_for_memory,
    str(uuid7()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)

print("\n===== 测试1：不带 query，仅枚举命名空间下全部条目（无向量检索） =====")
all_items = store.search(namespace_for_memory, limit=10)
for it in all_items:
    print(f"- key={it.key}, value={it.value}")

print("\n===== 测试2：带 query，语义向量检索（Ollama nomic-embed-text） =====")
semantic_result = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # Return top 3 matches
)
# 格式化打印，查看得分
for rank, it in enumerate(semantic_result):
    print(f"[{rank + 1}] score={it.score:.4f}, key={it.key}, value={it.value}")
