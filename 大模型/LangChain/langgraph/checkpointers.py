import json

from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State, config: RunnableConfig):
    checkpoint_ns(config)
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State, config: RunnableConfig):
    checkpoint_ns(config)
    return {"foo": "b", "bar": ["b"]}


def checkpoint_ns(config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    print(checkpoint_ns)


builder = StateGraph(State)
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(checkpointer=checkpointer, store=store)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
response = graph.invoke({"foo": "", "bar": []}, config)
print(response)

state = graph.get_state(config)
print('state:', state)

history = list(graph.get_state_history(config))

# Find the checkpoint before a specific node executed
before_node_b = next(s for s in history if s.next == ("node_b",))

# Find a checkpoint by step number
step_2 = next(s for s in history if s.metadata["step"] == 2)

# Find checkpoints created by update_state
forks = [s for s in history if s.metadata["source"] == "update"]

# Find the checkpoint where an interrupt occurred
interrupted = next(
    s for s in history
    if s.tasks and any(t.interrupts for t in s.tasks)
)



checkpoint_id = state.config["configurable"]["checkpoint_id"]

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_id}}
checkpoint_id_state = graph.get_state(config)
print(checkpoint_id_state)



# config=None returns checkpoints across every thread
for checkpoint_tuple in checkpointer.list(None):
    print(checkpoint_tuple.config["configurable"]["thread_id"])
