# 首先导入需要用到的数据包

from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import os


class BertClassifier(nn.Module):
    def __init__(self, bert_model, output_size):
        super(BertClassifier, self).__init__()
        self._bert = bert_model
        self._classifier = nn.Linear(bert_model.config.dim_feedforward, output_size)

    def forward(self, input_ids, attention_mask):
        # 获取BERT模型的CLS输出
        text_output = self._bert(input_ids=input_ids, attention_mask=attention_mask)
        # 得到线性层的结果
        logits = self._classifier(text_output.pooler_output)

        return logits


# 读取数据
class NewsDataSet(Dataset):
    def __init__(self, path):
        self._data = list()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                text, label = line.strip().split('\t')
                self._data.append((text, int(label)))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


# 用于dataloader,对于每个小批量的数据，进行分词和填充
def collate_fn(batch, tokenizer):
    texts = [text[0] for text in batch]  # 从batch数据中获取文本.
    labels = [text[1] for text in batch]  # 从batch数据中获取标签.
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # 标签变为张量.

    # 分词.
    tokens = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )

    return tokens['input_ids'], tokens['attention_mask'], labels_tensor


def train():
    # 读取数据集.
    news_data_set = NewsDataSet('bert-dataset/train.txt')
    print(len(news_data_set))

    # 加载bert模型.
    bert_model_path = r'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)  # 分词器.
    bert_model = BertModel.from_pretrained(bert_model_path)  # 加载模型.

    # dataset：要加载的数据集对象，必须是实现了len()和getitem()方法的对象
    data_loader = DataLoader(news_data_set,
                             batch_size=512,
                             shuffle=True,
                             collate_fn=lambda x: collate_fn(x, tokenizer))

    # 指定机器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device=', device)

    # 定义模型为10分类.
    model = BertClassifier(bert_model, output_size=10).to(device)
    model.train()
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    # 交叉熵损失误差
    criterion = nn.CrossEntropyLoss()
    # 存放模型
    os.makedirs('output_models', exist_ok=True)
    epoch_n = 10
    for epoch in range(1, epoch_n + 1):
        for batch_index, data in enumerate(data_loader):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            output = model(input_ids, attention_mask)
            loss = criterion(output, label)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数

            # 计算正确率，用于观察模型结果
            predict = torch.argmax(output, dim=1)
            correct = (predict == label).sum().item()
            acc = correct / output.size(0)
            print(f"Epoch {epoch}/{epoch_n}")  # 迭代轮数
            print(f"Batch {batch_index + 1}/{len(data_loader)}")
            print(f"Loss: {loss.item():.4f}")  # 损失
            print((f"Acc {correct}/{output.size(0)}=={acc:.3f}"))  # 正确率
            # 每一次迭代都保存一次模型结果
            model_name = f'./output_models/chinese_news_classify{epoch}.pth'
            print("saved model: %s" % (model_name))
            torch.save(model.state_dict(), model_name)


def predict():
    bert_model_path = r'bert-base-chinese'
    bert_model = BertModel.from_pretrained(bert_model_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    news_data_set = NewsDataSet('bert-dataset/test.txt')

    data_loader = DataLoader(news_data_set,
                             batch_size=1024,
                             shuffle=True,
                             collate_fn=lambda x: collate_fn(x, tokenizer))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(bert_model, 10)
    model.load_state_dict(torch.load('output_models/chinese_news_classify3.pth', map_location=device))
    model.to(torch.device(device))
    model.eval()

    acc = 0
    total_correct = 0
    total = 0
    for batch_index, data in enumerate(data_loader):
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        label = data[2].to(device)
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == label).sum().item()
            total_correct += correct
            total += outputs.size(0)
            print('total_correct=', total_correct, 'total=', total)
    acc = total_correct / total
    print(acc)


if __name__ == "__main__":
    train()
    print()
