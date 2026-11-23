
from pprint import pprint
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add


class CustomState(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: CustomState, config: RunnableConfig):
    print_checkpoint_ns(config)

    return {"foo": "a", "bar": ["a"]}


def node_b(state: CustomState, config: RunnableConfig):
    print_checkpoint_ns(config)

    return {"foo": "b", "bar": ["b"]}


def print_checkpoint_ns(config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    print(checkpoint_ns)


builder = StateGraph(CustomState)
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
response = graph.invoke({"foo": "", "bar": []}, config)
print('response=', response)


latest_state = graph.get_state(config)
pprint(latest_state)

history_state=graph.get_state_history(config)
pprint( list(history_state))