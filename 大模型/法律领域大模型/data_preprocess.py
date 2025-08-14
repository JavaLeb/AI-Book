import json

from pathlib import Path

path_str = "data/processed/chat_data.json"
tgt_file_path = Path(path_str)  # 替换为你的文件路径

if tgt_file_path.exists():
    tgt_file_path.unlink()  # 删除文件
    print(f"目标文件 {tgt_file_path} 已删除")
else:
    tgt_file_path.touch()  # 创建空文件
    print(f"目标文件 {tgt_file_path} 已创建")

# 打开原文件
src_path = 'data/raw/qa_data_train.json'
with open(src_path, "r", encoding="utf-8") as f:
    raw_datas = json.load(f)
    print(f'源文件{src_path}加载成功！')

# 将文件写到本地
with open(path_str, "w", encoding="utf-8") as f:
    for raw_data in raw_datas:
        raw_data["instruction"] = ""
        raw_data["output"] = raw_data["answer"]
        del raw_data["answer"]
        json.dump(raw_data, f, ensure_ascii=False)
        f.write('\n')
    print('数据预处理完成！')
