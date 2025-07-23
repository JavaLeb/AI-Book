import os
import platform
from transformers import AutoTokenizer, AutoModel


def load_model(eval=True):
    # 填写本地模型路径.
    model_path = r'E:\LLM\ZhipuAI\chatglm3-6b'
    # 加载分词器.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 加载模型.
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(4).cuda()
    # 将模型设置为评估模式
    if eval:
        model = model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# 根据系统设置清除历史记录命令
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

# 欢迎消息的提示文本
stop_command = 'stop'
welcome_prompt = f"欢迎使用，输入内容即可进行对话，清空对话历史请输入{clear_command}，终止程序请输入{stop_command}"


# 根据对话历史构建提示文本
def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\n大模型：{response}"
    return prompt


# 信号处理函数，用于处理程序终止
def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")  # 获取用户输入
        if query.strip() == stop_command:  # 终止.
            break
        elif query.strip() == clear_command:  # 清空历史.
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\n大模型：", end="")
        current_length = 0
        # response 模型生成的回复
        # history 历史对话
        # past_key_values 记录每个时间步的key和value，避免重复计算
        for response, history, past_key_values in model.stream_chat(
                tokenizer, query, history=history,
                past_key_values=past_key_values,
                return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()
