import torch
import matplotlib.pyplot as plt
from module import TransformerModel
from torch import nn
from data import Corpus, DataSetProcessor
import random
from sacrebleu.metrics import BLEU


class Meta(type):
    def __setattr__(cls, name, value):
        raise AttributeError("常量不可更改")


class Config(metaclass=Meta):
    """
    常量配置.
    """

    # 基础配置.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备.
    dtype = torch.float  # 数据类型.
    seed = 42  # 种子.

    # 数据配置.
    data_path = 'data/cmn-eng/cmn.txt'  # 文件路径.
    batch_size = 20  # 批大小.
    test_size = 5  # 测试数据的个数.
    num_line = 1000  # 读取文件的行数.
    train_size = 0.8  # 训练数据的比例.
    max_seq_len = 40  # 最大序列长度.
    epochs = 100  # 训练轮次.

    # 模型配置.
    d_model = 256  # 模型的维度.
    embedding_dim = d_model  # 词嵌入向量大小.
    num_head = 2  # 头数量.
    feedforward_dim = 512  # 隐藏层神经元个数.
    num_encoder_layers = 2  # encoder层数.
    num_decoder_layers = 2  # encoder层数.
    dropout_p = 0.1  # dropout概率

    # 优化器配置.
    lr = 0.0001  # 学习率.
    optimizer_betas = (0.9, 0.98)
    optimizer_eps = 1e-9


def plot_curve(epochs, train_loss, val_loss):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(10, 6))  # 设置图表大小
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='valid loss')
    plt.title('loss curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()


def transformer_main(trans_type):
    # 设置随机种子, 保证每次运行结果相同.
    torch.manual_seed(Config.seed)
    random.seed(Config.seed)

    # 数据处理.
    # 加载语料.
    corpus = Corpus(Config.data_path, device=Config.device,
                    num_line=Config.num_line, max_seq_len=Config.max_seq_len)
    dictionary = corpus.dictionary
    dataset_processor = DataSetProcessor(corpus.data_set)
    # 训练集和验证集划分.
    dataset_processor.split(train_size=Config.train_size, test_size=Config.test_size)
    # 数据集分批.
    train_loader, val_loader, test_loader = dataset_processor.batchify(Config.batch_size)

    # 定义模型.
    pad_id = dictionary.pad_id
    model = TransformerModel(num_embeddings=dictionary.size(), pad_id=pad_id,
                             config=Config(), transformer_type=trans_type).to(Config.device)

    # 优化器.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr,
        betas=Config.optimizer_betas, eps=Config.optimizer_eps)

    # 损失函数（忽略填充位置的损失）, 内部已经实现了softmax.
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 训练模型.
    model.fit(epochs=Config.epochs, train_loader=train_loader,
              val_loader=val_loader, optimizer=optimizer, loss_fn=loss_fn)

    model.load()  # 加载最佳模型进行预测.
    # 预测.
    for src_ids, tgt_ids, src_sent, tgt_sent in test_loader:
        outputs = model.predict(src_input=src_ids, key_token_ids=corpus.dictionary.key_token_ids)
        outputs = outputs.cpu().numpy().tolist()
        predict_sentences = []
        for output in outputs:  # 将输出转换为句子.
            out_sent = ''
            for token_id in output:
                token = corpus.dictionary.idx2token[token_id]
                if token == corpus.dictionary.eos_token:
                    continue
                out_sent += token
            out_sent = out_sent.lstrip(corpus.dictionary.bos_token)
            predict_sentences.append(out_sent)
        print(f"输入源序列: {src_sent}")
        print(f'目标序列: {tgt_sent}')
        print(f"预测序列: {predict_sentences}")

        bleu = BLEU(tokenize='zh')
        candidates = predict_sentences
        references = [[i] for i in tgt_sent]  # 元组变成列表的列表.
        score = bleu.corpus_score(candidates, references)
        print(f"BLEU 分数: {score.score: .2f}")
        print(f"详细信息: {score}")
    plot_curve(model.epochs, model.train_losses, model.val_losses)  # 绘制学习曲线.
