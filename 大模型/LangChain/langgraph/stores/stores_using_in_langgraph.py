from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from uuid_utils import uuid7

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from langgraph.models import qwen2dot5_7b_model as model

@dataclass
class Context:
    user_id: str


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def add_memory_node(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    memory_id = str(uuid7())
    memory = {"food_preference": "I like pizza"}
    runtime.store.put(namespace, memory_id, {"memory": memory})


builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node("add_memory_node", add_memory_node)
builder.add_edge(START, "add_memory_node")
builder.add_edge("add_memory_node", END)

checkpointer = InMemorySaver()
store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}

# First let's just say hi to the AI
for update in graph.stream(
        {"messages": [{"role": "user", "content": "hi"}]},
        config,
        stream_mode="updates",
        context=Context(user_id="1"),
):
    print(update)


async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    memory = {"food_preference": "My favorite drink is lemon tea"}
    memory_id = str(uuid7())

    await runtime.store.aput(namespace, memory_id, {"memory": memory})


async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    # Search based on the most recent message
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])
    print('info:', info)

    # ... Use memories in the model call
    model.invoke({"messages": [{"role": "user", "content": "my memories: " + info}]})


# Invoke the graph on a new thread
config = {"configurable": {"thread_id": "2"}}

# Let's say hi again
for update in graph.stream(
        {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]},
        config,
        stream_mode="updates",
        context=Context(user_id="1"),
):
    print(update)
