import torch


class Meta(type):
    def __setattr__(cls, name, value):
        raise AttributeError("不能修改常量")


class PytorchTransConfig(metaclass=Meta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备, 默认cuda, 可以设置为cpu.
    dtype = torch.float  # 数据类型.
    batch_size = 20  # 批大小.
    eval_batch_size = 10  # 评估时的批次大小.
    d_model = 256  # 模型的维度.
    embedding_dim = d_model  # 词嵌入向量大小.
    num_head = 4 # 头数量.
    dim_feedforward = 512  # 隐藏层神经元个数.
    num_encoder_layers = 4  # encoder层数.
    num_decoder_layers = 4  # encoder层数.
    dropout_p = 0.1  # dropout概率
    batch_interval = 10
    save = 'model.pt'  # 保存模型文件.
    dry_run = False
    data_path = 'data/cmn-eng/cmn.txt'  # 英译汉.
    num_line = 1000
    lr = 0.0001
    epochs = 200
    seed = 1111
    best_val_loss = None
    train_size = 0.8
    max_seq_len = 40
    optimizer_betas = (0.9, 0.98)
    optimizer_eps = 1e-9
