from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from model_identifiers import llama3dot2_3b_identifier


def my_print(step):
    print('+' * 50 + 'Response ' + str(step) + '+' * 50)


checkpointer = InMemorySaver()

agent = create_agent(
    model=llama3dot2_3b_identifier,
    middleware=[
        SummarizationMiddleware(
            model=llama3dot2_3b_identifier,
            trigger=("tokens", 200),
            keep=("messages", 3)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
response1 = agent.invoke({"messages": "hi, my name is bob"}, config)
my_print(step=1)
for message in response1["messages"]:
    message.pretty_print()
my_print(step=1)

response2 = agent.invoke({"messages": "write a short poem about cats"}, config)
my_print(step=2)
for message in response2["messages"]:
    message.pretty_print()
my_print(step=2)

response3 = agent.invoke({"messages": "now do the same but for dogs"}, config)
my_print(step=3)
for message in response3["messages"]:
    message.pretty_print()
my_print(step=3)

final_response = agent.invoke({"messages": "what's my name?"}, config)
my_print(step=4)
for message in final_response["messages"]:
    message.pretty_print()
my_print(step=4)
