import torch
from torch import nn, Tensor
import math
from module import Embedding, PositionalEncoding
from data import Corpus, DataSetProcessor
from config import PytorchTransConfig as Config
import time


class PytorchTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=512, dropout_p=0.1, max_seq_len=2000,
                 num_head=8, num_encoder_layers=6, number_decoder_layers=6, dim_feedforward=2048):
        super(PytorchTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        # 嵌入层.
        self.src_embedding = Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = Embedding(tgt_vocab_size, embedding_dim)
        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout_p=dropout_p, max_len=max_seq_len)
        # 内置transformer: 仅包含编码器和解码器.
        self.transformer = nn.Transformer(
            d_model=embedding_dim, nhead=num_head,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=number_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout_p,
            bias=False, batch_first=True  # 使用(batch, seq, feature)格式.
        )

        # 输出线性层.
        self.fc_out = nn.Linear(embedding_dim, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None, src_key_padding_mask: Tensor = None,
                tgt_key_padding_mask: Tensor = None) -> Tensor:
        """
        前向传播.
        :param src: 源序列.
        :param tgt: 目标序列.
        :param src_mask: 源蓄力注意力掩码.
        :param tgt_mask: 目标序列注意力掩码.
        :param src_key_padding_mask: 源序列键填充掩码.
        :param tgt_key_padding_mask: 目标序列键填充掩码.
        :return:
        """
        # 嵌入层 + 位置编码
        src_emb = self.src_embedding(src)
        src_emb = src_emb * math.sqrt(self.embedding_dim)
        src_pe_out = self.positional_encoding(src_emb)
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = tgt_emb * math.sqrt(self.embedding_dim)
        tgt_pe_out = self.positional_encoding(tgt_emb)

        # Transformer前向传播, 编码器-解码器层注意力使用了键填充掩码memory_key_padding_mask.
        outs = self.transformer(
            src=src_pe_out, tgt=tgt_pe_out,
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return self.fc_out(outs)

    # 为模型添加编码器方法
    def encode(self, src: Tensor, src_padding_mask: Tensor = None):
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.embedding_dim))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, memory_padding_mask: Tensor, tgt_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.embedding_dim))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_padding_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)


def create_attention_mask(tgt_input: Tensor) -> Tensor:
    sz = tgt_input.shape[1]

    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(Config.device)


def create_key_padding_mask(seq: Tensor, pad_id):
    key_padding_mask = (seq == pad_id)
    tmp = torch.zeros_like(key_padding_mask, dtype=torch.float)
    key_padding_mask = tmp.masked_fill(key_padding_mask, float('-inf'))

    return key_padding_mask.to(Config.device)


def compute_loss(src_input, tgt, loss_fn, pad_id):
    # 创建decoder的输入和输出.
    tgt_input = tgt[:, :-1]
    tgt_out = tgt[:, 1:]  # 排除序列开始的<bos>.

    # 创建掩码.
    src_padding_mask = create_key_padding_mask(src_input, pad_id)
    tgt_padding_mask = create_key_padding_mask(tgt_input, pad_id)
    tgt_mask = create_attention_mask(tgt_input)

    # 模型输出 shape:(batch_dim, seq_dim, vocab_dim).
    logits = model(
        src_input,
        tgt_input,
        src_mask=None,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_padding_mask,
        tgt_key_padding_mask=tgt_padding_mask
    )
    # 计算损失
    optimizer.zero_grad()
    # (batch_dim * seq_dim, vocab_dim).
    tgt_hat = logits.reshape(-1, logits.shape[-1])
    # (batch_dim * seq_dim,).
    tgt_true = tgt_out.reshape(-1)
    loss = loss_fn(tgt_hat, tgt_true)

    return loss


def train_epoch(epoch, model, train_loader, optimizer, loss_fn, pad_id):
    model.train()  # 开启模型训练模式.
    total_loss = 0
    batch_interval_loss = 0  # 批次区间损失.
    start_time = time.time()  # 开始时间.
    batch_num = 1  # 批次号.
    for src_input, tgt in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(src_input, tgt, loss_fn, pad_id)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # 累积每批损失.
        batch_interval_loss += loss.item()
        # if batch_num % Config.batch_interval == 0:  # 每个一些批次打印日志.
        #     batch_interval_loss = batch_interval_loss / Config.batch_interval  # 批次区间平均损失.
        #     elapsed = time.time() - start_time
        #     ppl = math.exp(batch_interval_loss)  # 困惑度（Perplexity，PPL）.
        #     time_per_batch = elapsed * 1000 / Config.batch_interval
        #     print(f'| epoch {epoch} | {batch_num}/{len(train_loader)} batches | lr {Config.lr} '
        #           f'| ms/batch {time_per_batch} | loss {batch_interval_loss} | ppl {ppl}')
        #     batch_interval_loss = 0
        #     start_time = time.time()
        batch_num += 1

    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, val_loader, loss_fn, pad_id):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src_input, tgt in val_loader:
            loss = compute_loss(src_input, tgt, loss_fn, pad_id)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# 预测函数 (贪婪解码)
def greedy_decode(model, src: Tensor, src_padding_mask: Tensor, dictionary):
    model.eval()
    device = Config.device
    # 编码器输出
    memory = model.encode(src, src_padding_mask)
    # 初始化解码器输入 (起始符号) <bos>.
    ys = torch.ones(src.size(0), 1).fill_(dictionary.bos_id).type_as(src).to(device)
    eos_flags = torch.zeros(ys.size(0), dtype=torch.bool).to(device)  # 终止标志

    for i in range(Config.max_seq_len - 1):
        tgt_mask = create_attention_mask(ys).type_as(src_padding_mask).to(device)
        tgt_padding_mask = (ys == dictionary.pad_id)
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
        eos_flags |= (next_word == dictionary.eos_id)
        # 所有序列遇到EOS则终止
        if eos_flags.all():
            break

    return ys


if __name__ == '__main__':

    torch.manual_seed(42)

    # 数据处理.
    corpus = Corpus(Config.data_path, device=Config.device,
                    num_line=Config.num_line, max_seq_len=Config.max_seq_len)
    dataset_processor = DataSetProcessor(corpus.data_set)
    train_dataset, val_dataset = dataset_processor.split(Config.train_size)
    train_loader, val_loader = dataset_processor.batchify(Config.batch_size)

    # 定义模型.
    vocab_size = corpus.dictionary.size()
    model = PytorchTransformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
                               embedding_dim=Config.embedding_dim, dropout_p=Config.dropout_p,
                               max_seq_len=Config.max_seq_len, num_head=Config.num_head,
                               num_encoder_layers=Config.num_encoder_layers,
                               number_decoder_layers=Config.num_decoder_layers,
                               dim_feedforward=Config.dim_feedforward).to(Config.device)
    # 损失函数（忽略填充位置的损失）, 内部已经实现了softmax.
    pad_id = corpus.dictionary.pad_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    # 优化器.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr,
        betas=Config.optimizer_betas, eps=Config.optimizer_eps)

    # 训练循环
    best_val_loss = float('inf')
    best_model_name = 'best_translation_model.pt'
    for epoch in range(Config.epochs):
        # 每个样本的平均损失.
        train_loss = train_epoch(epoch, model, train_loader, optimizer, loss_fn, pad_id=pad_id)
        val_loss = evaluate(model, val_loader, loss_fn, pad_id=pad_id)
        print(f'Epoch [{epoch + 1}/{Config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, best_model_name)
    print(f"训练完成！模型已保存为 '{best_model_name}'")

    # 加载最佳模型进行预测
    print("加载模型进行预测...")
    checkpoint = torch.load(best_model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取一个测试样本
    src, tgt = train_loader.dataset[0]
    src, tgt = src.reshape(1, -1), tgt.reshape(1, -1)
    src_ids = src.cpu().numpy().tolist()
    src_tokens = corpus.dictionary.ids2tokens(src_ids[0])
    tgt_tokens = corpus.dictionary.ids2tokens(tgt[0])
    # 生成源序列填充掩码.
    src_padding_mask = create_key_padding_mask(src, pad_id)
    # 进行预测
    output = greedy_decode(model, src, src_padding_mask, corpus.dictionary)
    output_ids = output.cpu().numpy().tolist()
    output_tokens = corpus.dictionary.ids2tokens(output_ids[0])
    print(f"输入源序列: {src_tokens})")
    print(f'目标序列: {tgt_tokens}')
    print(f"预测序列: {output_tokens}")
