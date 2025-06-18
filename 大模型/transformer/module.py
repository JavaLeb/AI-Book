import torch
from torch import nn
import math


class Embedding(nn.Module):
    """
    嵌入层 (直接封装内置实现).
    """

    def __init__(self, num_embeddings, embedding_dim):
        """
        初始化方法.
        embedding矩阵的第i(i=0,...)行表示索引为i的token的词嵌入.
        :param num_embeddings: embedding的token数量.
        :param embedding_dim: embedding维度.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, X):
        """
        根据索引查找词嵌入.
        :param X:  X的值表示token的索引, shape=(batch_size, seq_len).
        :return:
        """
        return self.embedding(X) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    r"""位置编码层，获取序列中token之间的相对或绝对位置信息.
        位置编码向量与embedding向量维度相同, 因此它们可以求和.
        使用不同频率的sine和cosine函数.
    """

    def __init__(self, embedding_dim, dropout_p=0.1, max_len=5000):
        """
        位置编码层.
        假设embedding_dim=n,一个位置pos对应一个位置向量p_pos=(sin(w0*pos),cos(w1*pos),...,sin(wn-2*pos),cos(wn-1*pos)).
        w_2i+1=w_2i.

        :param embedding_dim: embedding向量的维度大小.
        :param dropout_p: dropout值，默认0.1.
        :param max_len: 输入序列最大长度.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)  # 初始化Dropout.
        # pe: shape=(pos,embedding_dim), max_len用于生产足够长的位置向量.
        pe = torch.zeros(max_len, embedding_dim)
        # unsqueeze在指定维度插入长度为1的新维度,例如shape=(3,)，当dim=1，则变为shape=(3,1).
        # position: shape=(pos,1).
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)  # 生成足够长的位置pos.
        # 相当于公式中的w_2i, 使用exp避免精度丢失, shape=(embedding_dim/2,).
        w_2i = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(w_2i * position)  # 所有pos的位置向量的偶数位置使用正弦函数.
        pe[:, 1::2] = torch.cos(w_2i * position)  # 所有pos的位置向量奇数位置使用余弦函数.
        # 初始pe的形状(pos,embedding_dim), 然后在dim=0插入批次维度变为(1,pos,embedding_dim).
        pe = pe.unsqueeze(dim=0)
        # 注册, 调用to()方法时, 可迁移计算device.
        self.register_buffer('pe', pe)

    def forward(self, X):
        r"""
        输出使用dropout.
        :param X: 输入 [batch_size, seq_len, embedding_dim].
        形状:
            输入: [batch_size, seq_len, embedding_dim]
            输出: [batch_size, seq_len, embedding_dim]
        """
        # X.size(1)获取序列长度, self.pe[:,:x.size(1), :]的shape(1,seq_len,embedding_dim).
        p = self.pe[:, :X.size(1), :]
        # 通过在批次维度上广播与X 求和.
        X = X + p

        # 以某个概率将x的值变为0, 保留的值乘以1/p, 保持输出的值的期望不变.
        # 对于某个元素x, E[output] = p * 0 + (1-p) * (1/(1-p)) * x = x.

        return self.dropout(X)
