from sentence_transformers import SentenceTransformer, util
import streamlit as st

st.set_page_config(
    page_title="智能客服系统",
    page_icon=":robot:",
    layout="wide"
)

# 处理数据
with open("data/item_des.txt", 'r', encoding='utf-8') as f:
    item_data = f.readlines()  # 按行读取.

# 将对应的产品描述和产品名结合
item_dict = {}
for data in item_data:
    des_list = data.rstrip().split('\t')  # 按tab键分割为key和value形式.
    if des_list[0] in item_dict:  # key: [value1,value2].
        item_dict[des_list[0]].append(des_list[1])
    else:
        item_dict[des_list[0]] = [des_list[1]]
# key列表.
key_list = [key for key in item_dict]

# 加载embedding模型, 并将每个产品名称转换为一个向量.
embedding_model = SentenceTransformer(r'E:\pycharm_workspace\llm\m3e-small')
embeddings = embedding_model.encode(key_list)

# 初始化历史记录
# 这里需要初始化两个历史记录，history是用于和大模型交互的历史记录，这里面包含一些提示，不适合展示给用户
# show_history里存储着用户和智能客服一问一答，适合展示给用户
if "history" not in st.session_state:
    st.session_state.history = []
if "show_history" not in st.session_state:
    st.session_state.show_history = []

# 设定清理会话历史按钮
buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.show_history = []
    st.rerun()

# 在页面展示聊天内容
for i, message in enumerate(st.session_state.show_history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

# 获取用户输入
prompt_text = st.chat_input("请输入您的问题")

from load_model import tokenizer, model

# 如果用户输入了内容,则生成回复
if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    show_history = st.session_state.show_history
    show_history.append(
        {'role': 'user', 'content': prompt_text}
    )
    # 识别用户想要查询的物体
    recognize_prompt = f"""
    请识别用三个反引号括起来的文本中，用户想要查询什么。如果识别成功，只需要输出识别的商品，如果识别失败，则输出识别失败。
    ```{prompt_text}```
    """
    res, _ = model.chat(tokenizer, recognize_prompt, history=[],
                        top_p=0.7,
                        temperature=0.95,
                        max_length=4095)
    print('识别响应res=', res)
    recognize_resp = res.replace("```", "")
    print('识别响应recognize_resp=', recognize_resp)
    # 如果识别失败，则返回无法识别你需要查询的商品。
    if "识别失败" in recognize_resp:
        response = "无法识别你需要查询的商品。"
        message_placeholder.markdown(response)
        history.append(
            {'role': 'user', 'content': recognize_prompt}
        )
        show = {"role": "assistant", "content": response}
        history.append(show)
        show_history.append(show)
    else:
        # 若识别成功，则通过相似度计算找到目标商品
        item_embedding = embedding_model.encode(recognize_resp)
        sim_score = util.pytorch_cos_sim(embeddings, item_embedding)
        print('sim_score.max=', sim_score.max())
        # 若识别出的商品不存在
        if sim_score.max() < 0.6:
            res_rs = recognize_resp.rstrip('。')
            response = f'非常抱歉，我无法为您提供关于 {res_rs} 的信息。'
        else:
            target_item = key_list[sim_score.argmax()]
            target_item_des = '\n'.join(item_dict[target_item])
            # 将对话记录存储
            history.append(
                {"role": "system",
                 "content": f"你是一位智能客服，你要热情且谦逊地回答用户的问题。下面用三个反引号括起来的是关于{target_item}的一些商品描述，你的回答要基于这些描述，不可捏造。```{target_item_des}```"}
            )
            history.append(
                {"role": "user", "content": prompt_text}
            )
            # 得到回复
            res, _ = model.chat(tokenizer, prompt_text, history=history,
                                top_p=0.7,
                                temperature=0.95,
                                max_length=4095)

            response = res.replace("```", "")
        history.append({"role": "assistant", "content": response})
        show_history.append({"role": "assistant", "content": response})
        message_placeholder.markdown(response)
    # 更新历史记录
    st.session_state.history = history
    st.session_state.show_history = show_history
