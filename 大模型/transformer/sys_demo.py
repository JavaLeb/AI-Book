import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Transformer
import math
from typing import Tuple
from user_transformer import UserTransformer

# 设置超参数
SRC_VOCAB_SIZE = 8  # 源语言词汇表大小
TGT_VOCAB_SIZE = 10  # 目标语言词汇表大小
EMB_SIZE = 2  # 嵌入维度
NHEAD = 2  # 注意力头数
FFN_HID_DIM = 2  # 前馈网络维度
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
MAX_LEN = 6  # 最大序列长度
DROPOUT_P = 0.0

# 特殊符号索引
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


# 构建Transformer模型
class Seq2SeqTransformer(nn.Module):
    def __init__(self, transformer_type='sys'):
        super(Seq2SeqTransformer, self).__init__()

        if transformer_type == 'sys':
            # 共享的嵌入层
            self.src_embedding = nn.Embedding(SRC_VOCAB_SIZE, EMB_SIZE)
            self.tgt_embedding = nn.Embedding(TGT_VOCAB_SIZE, EMB_SIZE)

            # 位置编码
            self.positional_encoding = PositionalEncoding(EMB_SIZE, dropout=DROPOUT_P, max_len=MAX_LEN)

            self.transformer = Transformer(
                d_model=EMB_SIZE,
                nhead=NHEAD,
                num_encoder_layers=NUM_ENCODER_LAYERS,
                num_decoder_layers=NUM_DECODER_LAYERS,
                dim_feedforward=FFN_HID_DIM,
                dropout=DROPOUT_P,
                bias=False,
                batch_first=True  # 使用(batch, seq, feature)格式
            )

        # 输出线性层
        self.fc_out = nn.Linear(EMB_SIZE, TGT_VOCAB_SIZE)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_key_padding_mask: Tensor,
                tgt_key_padding_mask: Tensor) -> Tensor:
        # 嵌入层 + 位置编码
        src_emb = self.src_embedding(src)
        src_pe = self.positional_encoding(src_emb * math.sqrt(EMB_SIZE))
        tgt_emb = self.tgt_embedding(tgt)
        tgt_pe = self.positional_encoding(tgt_emb * math.sqrt(EMB_SIZE))

        # Transformer前向传播.
        outs = self.transformer(
            src_pe,
            tgt_pe,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return self.fc_out(outs)

    # 为模型添加编码器方法
    def encode(self, src: Tensor, src_padding_mask: Tensor = None):
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(EMB_SIZE))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_padding_mask: Tensor, tgt_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(EMB_SIZE))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_padding_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)


# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 生成掩码
def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def create_mask(src: Tensor, tgt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

    # 填充掩码
    src_padding_mask = (src == 0)  # 假设0是<pad>
    tgt_padding_mask = (tgt == 0)

    return tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)


# 训练函数
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for src, tgt in train_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 创建decoder的输入和输出
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # 创建掩码
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # 模型输出
        logits = model(
            src,
            tgt_input,
            src_mask=None,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # 计算损失
        optimizer.zero_grad()
        pred_hat = logits.reshape(-1, logits.shape[-1])
        pred_true = tgt_out.reshape(-1)
        loss = loss_fn(pred_hat, pred_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(
                src,
                tgt_input,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)


# 预测函数 (贪婪解码)
def greedy_decode(model, src: Tensor, src_padding_mask: Tensor, max_len: int, start_symbol: int, device):
    model.eval()
    src = src.to(device)

    # 编码器输出
    memory = model.encode(src, src_padding_mask)

    # 初始化解码器输入 (起始符号) <bos>.
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src).to(device)
    eos_flags = torch.zeros(ys.size(0), dtype=torch.bool).to(device)  # 终止标志

    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).type_as(src_padding_mask).to(device)
        tgt_padding_mask = (ys == PAD_IDX)
        # 解码器输出
        out = model.decode(ys, memory, tgt_mask, memory_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_padding_mask)

        # 预测下一个token
        prob = model.fc_out(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)

        # 添加到序列
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

        # 如果遇到结束符则停止
        # 更新终止标志
        eos_flags |= (next_word == EOS_IDX)
        # 所有序列遇到EOS则终止
        if eos_flags.all():
            break

    return ys.squeeze(0)


# 模拟数据集类
class TranslationDataset(Dataset):
    def __init__(self, num_samples=5, src_len=1, tgt_len=2, max_len=MAX_LEN):
        self.data = []
        for _ in range(num_samples):
            # 生成随机序列 (模拟语言数据)
            src = torch.randint(3, SRC_VOCAB_SIZE - 1, (src_len,))
            tgt = torch.randint(3, TGT_VOCAB_SIZE - 1, (tgt_len,))

            # 添加开始/结束标记
            src = torch.cat([torch.tensor([BOS_IDX]), src, torch.tensor([EOS_IDX])])
            tgt = torch.cat([torch.tensor([BOS_IDX]), tgt, torch.tensor([EOS_IDX])])

            # 填充到相同长度
            src = nn.functional.pad(src, (0, max_len - len(src)), value=PAD_IDX)
            tgt = nn.functional.pad(tgt, (0, max_len - len(tgt)), value=PAD_IDX)

            self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 主函数
if __name__ == "__main__":
    # 选择设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = 'cpu'

    torch.manual_seed(42)
    # 准备数据
    dataset = TranslationDataset(num_samples=5)

    # 训练集和测试集划分.
    train_size = int(0.4 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    torch.manual_seed(42)
    # 初始化模型
    model_type = 'sys'
    # model_type = 'user'

    if model_type == 'sys':
        model = Seq2SeqTransformer().to(device)
    else:
        model = UserTransformer(src_num_embeddings=SRC_VOCAB_SIZE, tgt_num_embeddings=TGT_VOCAB_SIZE,
                                embedding_dim=EMB_SIZE, num_heads=NHEAD,
                                feedforward_dim=FFN_HID_DIM, num_encoder_layers=NUM_ENCODER_LAYERS,
                                num_decoder_layers=NUM_DECODER_LAYERS, dropout_p=DROPOUT_P).to(device)

    # 损失函数（忽略填充位置的损失）
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_translation_model.pt')

    print("训练完成！模型已保存为 'best_translation_model.pt'")

    # 加载最佳模型进行预测
    print("\n加载模型进行预测...")
    checkpoint = torch.load('best_translation_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取一个测试样本
    for src, tgt in train_loader:
        test_sample = src

        # 生成源掩码
        src_padding_mask = (test_sample == PAD_IDX).to(device)

        # 进行预测
        output = greedy_decode(model, test_sample, src_padding_mask, MAX_LEN, BOS_IDX, device)

        print(f"输入序列: {test_sample.cpu().numpy().tolist()}")
        print(f"预测序列: {output.cpu().numpy().tolist()}")
