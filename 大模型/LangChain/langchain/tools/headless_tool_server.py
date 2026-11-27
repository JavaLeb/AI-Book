import sqlite3
from typing import Any

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from pydantic import BaseModel


# ---------- 步骤 1：schema-only headless tool ----------
class SendEmailInput(BaseModel):
    """to: 邮件接收人. subject: 主题. body: 正文."""
    to: str
    subject: str
    body: str


@tool(
    "send_email",
    description="Send an email. Execution is deferred to an external backend worker.",
    args_schema=SendEmailInput,
)
def send_email(to: str, subject: str, body: str, runtime: ToolRuntime) -> Any:
    # 不执行真实动作，只抛出结构化中断负载
    payload = {
        "type": "tool",
        "tool_call": {
            "id": runtime.tool_call_id,
            "name": "send_email",
            "args": {"to": to, "subject": subject, "body": body},
        },
    }

    return interrupt(payload)


# ---------- 步骤 2：持久化 checkpointer + agent ----------

DB_PATH = "checkpoints.db"
# SQLite 需要 check_same_thread=False，因为 FastAPI 可能在不同线程执行同步端点
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(
    model="ollama:qwen2.5:7b-instruct",
    tools=[send_email],
    checkpointer=checkpointer,
)


from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command

app = FastAPI(title="Headless Tool FastAPI Demo")


class RunRequest(BaseModel):
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    tool_result: dict


@app.post("/run")
def run(req: RunRequest):
    """
    第一次 invoke：如果模型调用 send_email，图会在 interrupt() 处暂停。
    状态已由 SqliteSaver 持久化，HTTP 请求可以安全返回。
    """

    thread_id = req.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )

    interrupts = result.get("__interrupt__")
    if not interrupts:
        return {
            "status": "completed",
            "thread_id": thread_id,
            "messages": [
                {"role": m.type, "content": m.content}
                for m in result.get("messages", [])
            ],
        }

    interrupt_obj = interrupts[0]
    return {
        "status": "interrupted",
        "thread_id": thread_id,
        "payload": interrupt_obj.value,
    }


@app.post("/resume")
def resume(req: ResumeRequest):
    """
    第二次 invoke：在服务端恢复图。resume 值会成为 interrupt() 的返回值。
    必须使用和 /run 相同的 thread_id，才能找到同一份 checkpoint。
    """
    config = {"configurable": {"thread_id": req.thread_id}}

    final_result = agent.invoke(
        Command(resume=req.tool_result),
        config=config,
    )

    return {
        "status": "completed",
        "thread_id": req.thread_id,
        "messages": [
            {"role": m.type, "content": m.content}
            for m in final_result.get("messages", [])
        ],
    }
