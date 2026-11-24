from models import qwen3_VL_2b_model as model
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage


def image_to_data_url(image_path: str) -> str:
    """把本地图片编码成 Ollama 可识别的 base64 data URL。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    mime = mime_map.get(path.suffix.lower().lstrip("."), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# 2. 把猫图片编码为 data URL
image_url = image_to_data_url("../../data/cat.png")   # ← 换成你的猫图片路径

# 3. 构建多模态消息
message = HumanMessage(
    content=[
        {"type": "text", "text": "请详细描述这张图片中的内容。"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
)

# 4. 调用模型并打印描述
response = model.invoke([message])

print("---- 模型描述 ----")
print(response.content)

# 如果 content_blocks 有内容，也可以查看结构化块
if response.content_blocks:
    print("\n---- content_blocks ----")
    print(response.content_blocks)