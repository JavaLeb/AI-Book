from langchain.agents import create_agent
from langchain_agentsearch import AgentSearchTool, AgentSearchExtractTool, AgentSearchRenderTool
from models import qwen2dot5_7b_model as model
tools = [
    AgentSearchTool(mode="free"),   # search stays free, even after you add a wallet
    AgentSearchExtractTool(),       # read a URL as markdown
    AgentSearchRenderTool(),        # only for JavaScript-heavy pages
]

SYSTEM = """You answer questions from the live web.
1. Call agentsearch_search first to find candidate URLs.
2. Pick at most 2 results that look authoritative and call agentsearch_extract on them
   with max_chars=8000. Do not extract every result.
3. If extract returns empty or very short content, call agentsearch_render on that URL once.
4. Answer from the page text and cite every URL you used.
5. If a tool returns an error, say what you could not read and answer from what you have."""

agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM)

res = agent.invoke({"messages": [{"role": "user",
    "content": "What changed in the latest LangChain release? Cite URLs."}]})
print(res["messages"][-1].content)