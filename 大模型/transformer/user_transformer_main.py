from torch import nn
import torch
import math
from data import Corpus, get_batch, batchify
import time
from user_transformer import UserTransformer


class Config:
    """
    配置.
    """

    def __init__(self):
        self.device = 'cuda'  # 设备, 默认cuda, 可以设置为cpu.
        self.dtype = torch.float

        self.batch_size = 2  # 批大小.
        self.eval_batch_size = 10  # 评估时的批次大小.
        self.d_model = 2
        self.embedding_dim = self.d_model  # 词嵌入向量大小.
        self.nhead = 2  # 头数量.
        self.dim_feedforward = 2  # 隐藏层神经元个数.
        self.num_encoder_layers = 2  # encoder层数.
        self.num_decoder_layers = 2  # encoder层数.
        self.dropout_p = 0.1  # dropout概率
        self.seq_len = 3  # 序列长度.
        self.clip = 0.25  # 梯度裁剪.
        self.log_interval = 200
        self.save = 'model.pt'  # 保存模型文件.
        self.dry_run = False
        self.data_path = 'data/wikitext-2'  # 路径.
        self.lr = 5.0
        self.epochs = 6
        self.seed = 1111
        self.best_val_loss = None


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, config.seq_len):
            data, targets = get_batch(data_source, i, config.seq_len)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train_epoch(epoch, criterion):
    model.train()  # 开启训练模式, 训练模型会启动dropout.
    total_loss = 0  # 总损失.
    start_time = time.time()  # 开始时间.
    for batch, i in enumerate(range(0, train_data.size(0) - 1, config.seq_len)):
        src_data, target_data = get_batch(train_data, i, config.seq_len)
        targets = target_data.reshape(-1)
        model.zero_grad()  # 梯度清零.
        output = model(src_data, target_data)  # 计算输出 shape(batch_size,seq_len,voc_size).
        output = output.view(-1, n_tokens)  # shape(batch_size * seq_len, voc_size).
        output = torch.softmax(output, dim=-1)
        output = torch.argmax(output, dim=-1)

        loss = criterion(output, targets)  # 计算损失, 索引看成是类别编码.
        loss.backward()  # 计算梯度.

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-config.lr)

        total_loss += loss.item()
        if batch % config.log_interval == 0 and batch > 0:
            cur_loss = total_loss / config.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // config.seq_len, config.lr,
                              elapsed * 1000 / config.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if config.dry_run:
            break


def start(criterion):
    try:
        for epoch in range(1, config.epochs + 1):
            epoch_start_time = time.time()
            train_epoch(epoch, criterion)
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not config.best_val_loss or val_loss < config.best_val_loss:
                with open(config.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                config.lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    config = Config()  # 创建配置.
    torch.manual_seed(config.seed)
    # 加载语料.
    corpus = Corpus(config.data_path)

    train_data = batchify(corpus.train, config.batch_size, device=config.device)
    val_data = batchify(corpus.valid, config.eval_batch_size, device=config.device)
    test_data = batchify(corpus.test, config.eval_batch_size, device=config.device)

    n_tokens = corpus.get_dic_token_num()  # 获取字典token数量.
    # 初始化Transformer模型.
    model = UserTransformer(num_embeddings=n_tokens, embedding_dim=config.embedding_dim, num_heads=config.nhead,
                        feedforward_dim=config.dim_feedforward, num_encoder_layers=config.num_encoder_layers,
                        num_decoder_layers=config.num_decoder_layers, dropout_p=config.dropout_p).to(
        device=config.device, dtype=config.dtype)
    criterion = nn.CrossEntropyLoss()  # 损失函数.
    start(criterion)
