import requests
import uuid
BASE_URL = "http://localhost:8000"
RUN_URL = f"{BASE_URL}/run"
RESUME_URL = f"{BASE_URL}/resume"

def send_email(
    payload
):
    print(f"[worker] 执行真实动作：{payload}")

def main():
    # 第一步：启动 agent
    thread_id = str(uuid.uuid4())
    run_resp = requests.post(
        RUN_URL,
        json={
            "message": "给 alice@example.com 发邮件，主题“会议改期”，正文“改到下午三点”。",
            "thread_id": thread_id,
        },
    )
    print(f"run_resp={run_resp}")
    run_data = run_resp.json()
    print(f"[worker] /run 返回：{run_data}")

    if run_data["status"] != "interrupted":
        print("[worker] 没有中断，流程结束。")
        return

    # 第二步：解析中断 payload
    payload = run_data["payload"]
    tool_call = payload["tool_call"]
    args = tool_call["args"]

    print(f"\n[worker] 收到工具调用：{tool_call['name']}")
    print(f"[worker] 参数：{args}")
    send_email(payload)

    # 真实场景：SMTP / 邮件微服务 / 浏览器 API / 人工审批
    tool_result = {
        "status": "sent",
        "to": args["to"],
        "subject": args["subject"],
        "message_id": f"demo-{tool_call['id']}",
    }
    print(f"[worker] 执行完成，结果：{tool_result}")

    # 第三步：把结果送回服务端，恢复图
    resume_resp = requests.post(
        RESUME_URL,
        json={
            "thread_id": thread_id,
            "tool_result": tool_result,
        },
    )
    resume_data = resume_resp.json()
    print("\n[worker] /resume 返回：")
    print(resume_data)


if __name__ == "__main__":
    main()