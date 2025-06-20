import torch
from torch import nn
import math
from torch import Tensor


class Masker:
    """
    掩码器.
    """

    def __init__(self, padding_id, device=None):
        self.neg_inf = float('-inf')  # 负无穷大.
        self.padding_id = padding_id
        self.device = device

    def create_future_mask(self, inputs: Tensor) -> Tensor:
        """
        创建注意力掩码.
        例如：如果inputs的shape(2,3), 返回(3,3)矩阵
                [[0, -inf, -inf],
                 [0, 0,    -inf],
                 [0, 0,    0]]

        :param inputs: shape(batch_dim, seq_dim).
        :return: 倒三角: shape(seq_dim, seq_dim).
        """
        seq_len = inputs.shape[1]
        full_neg_inf = torch.full((seq_len, seq_len), self.neg_inf)
        # 对角线diagonal=1以及之上的元素保留, 其余元素设置为0.
        return torch.triu(full_neg_inf, diagonal=1).to(self.device)

    def create_key_padding_mask(self, inputs: Tensor):
        """
        创建键填充掩码.
        例如: inputs=[[1,4,0],[5,0,0]], pad_id=0, 则返回掩码如下:
        [[0, 0,    -inf],
         [0, -inf, -inf]].

        :param inputs: 输入序列 shape=(batch_dim, seq_dim).
        :return: 键填充掩码: shape=(batch_dim, seq_dim).
        """

        key_padding_mask = (inputs == self.padding_id)  # 布尔型掩码.
        tmp = torch.zeros_like(key_padding_mask, dtype=torch.float)
        key_padding_mask = tmp.masked_fill(key_padding_mask, self.neg_inf)  # 浮点型掩码.

        return key_padding_mask.to(self.device)


class Embedding(nn.Module):
    """
    嵌入层: 封装内置实现.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int = 512):
        """
        初始化方法.
        embedding矩阵的第i(i=0,...)行表示索引为i的token的嵌入向量.
        :param num_embeddings: embedding的token数量.
        :param embedding_dim: embedding维度.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, X):
        """
        根据索引查找词嵌入.
        :param X:  X的每个元素表示token的索引, shape=(batch_dim, seq_dim).
        :return: X对应的词嵌入向量, shape=(batch_dim, seq_dim, embedding_dim).
        """
        return self.embedding(X)


class Dropout(nn.Module):
    """
    封装内置Dropout: 方便训练模型与非训练模式切换.
    """

    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        """
            以某个概率将X的元素x的值变为0, 保留的值乘以1/p, 保持输出的值的期望不变.
            对于某个元素x, E[output] = p * 0 + (1-p) * (1/(1-p)) * x = x.
        :param X:
        :return:
        """
        if self.training:  # 训练模型.
            X = self.dropout(X)

        return X


class PositionalEncoding(nn.Module):
    """位置编码层，获取序列中token之间的相对或绝对位置信息.
        位置编码向量与embedding向量维度相同, 因此它们可以求和.
        使用不同频率的sine和cosine函数.
    """

    def __init__(self, pe_dim=512, dropout_p=0.1, max_seq_len=5000):
        """
        假设embedding_dim=n,一个位置pos对应一个位置向量p_pos=(sin(w0*pos),cos(w1*pos),...,sin(wn-2*pos),cos(wn-1*pos)).
        w_2i+1=w_2i.

        :param pe_dim: 位置编码向量的维度大小.
        :param dropout_p: dropout概率, 默认0.1.
        :param max_seq_len: 输入序列最大长度.
        """
        super().__init__()
        self.dropout = Dropout(dropout_p)  # 初始化Dropout.
        # pe: shape=(pos_dim,pe_dim).
        pe = torch.zeros(max_seq_len, pe_dim)
        # unsqueeze在指定维度插入长度为1的新维度,例如shape=(3,)，当dim=1，则变为shape=(3,1).
        # position: shape=(pos_dim,1).
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(dim=1)  # 生成足够长的位置pos.
        # 相当于公式中的w_2i, 使用exp避免精度丢失, shape=(pe_dim/2,).
        w_2i = torch.exp(
            torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim))

        pe[:, 0::2] = torch.sin(w_2i * position)  # 所有pos的位置向量的偶数位置使用正弦函数.
        pe[:, 1::2] = torch.cos(w_2i * position)  # 所有pos的位置向量奇数位置使用余弦函数.
        # 初始pe的形状(pos_dim,pe_dim), 然后在dim=0插入批次维度变为(1,pos_dim,pe_dim).
        pe = pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe)  # 注册, 模型定义时调用to(device)方法, 可迁移到device上.

    def forward(self, X):
        r"""
        输出使用dropout.

        :param X: 输入, shape=(batch_dim, seq_dim, embedding_dim).
        :return 输出, shape=(batch_dim, seq_dim, embedding_dim).
        """
        # X.size(1)为序列长度, p的shape=(1,seq_dim,embedding_dim).
        p = self.pe[:, :X.size(1), :]
        X = X + p  # 通过在批次维度上广播与X 求和.

        return self.dropout(X)


class AddNorm(nn.Module):
    """
    层标准化, 默认按照最后一个维度标准化.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, layer_input, layer_out):
        """

        :param layer_input: 层的输入.
        :param layer_out: 层的输出。
        :return:
        """
        add = layer_input + layer_out  # 残差连接.

        return self.layer_norm(add)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力.
    """

    def __init__(self, head_dim):
        """
        初始化.
        :param head_dim: 每个头的维度.
        """

        super().__init__()
        self.head_dim = head_dim

    def forward(self, key, query, value, attn_mask):
        """
        前向传播.
        :param key: 键, shape=(batch_dim, src_seq_len, key_dim).
        :param query: 查询, shape=(batch_dim, tgt_seq_len, query_dim).
        :param value: 值, shape=(batch_dim, src_seq_len, value_dim).
        :param attn_mask: 注意力掩码, shape=(batch_dim,num_heads,src_seq_len,tgt_seq_len).
        :return: h: shape=(batch_dim, num_heads, src_seq_len, head_dim)
        """
        # qk : (batch_dim, num_heads, src_seq_len, tgt_seq_len).
        qk = query @ torch.transpose(key, -1, -2)  # 可使用torch.bmm()批量矩阵乘法.
        # scaled_qk: (batch_dim, num_heads, src_seq_len, tgt_seq_len).
        scaled_qk = qk / math.sqrt(self.head_dim)
        scaled_qk += attn_mask
        # attention_weights: (batch_dim, num_heads, src_seq_len, tgt_seq_len).
        attention_weights = torch.softmax(scaled_qk, dim=-1)
        # h: (batch_dim, num_heads, src_seq_len, head_dim).
        h = attention_weights @ value  # Pytorch内置实现在权重上使用了Dropout技术.

        return h


class MultiHeadAttention(nn.Module):
    """
    多头注意力层.
    """

    def __init__(self, qkv_input_dim: int = 512, num_heads: int = 8,
                 query_dim: int = 512, key_dim: int = 512, value_dim: int = 512) -> None:
        """
        初始化方法.
        :param qkv_input_dim: 多头注意力qkv输入的维度n.
        :param num_heads: 头的数量h.
        :param query_dim: 查询的维度d_k.
        :param key_dim: 键的维度d_k.
        :param value_dim: 值的维度d_v.
        """
        super().__init__()

        self.qkv_input_dim = qkv_input_dim  # n.
        self.num_heads = num_heads  # h.
        self.query_dim = query_dim  # d_k.
        self.key_dim = key_dim  # d_k.
        self.value_dim = value_dim  # d_v.
        self.head_dim = qkv_input_dim // num_heads  # 每个头的维度.

        assert self.head_dim * num_heads == qkv_input_dim  # 需要整除.

        self.dot_product_attention = ScaledDotProductAttention(self.head_dim)

        # 查询、键和值的权重初始化. 线性模型权重w的shape=(out_features,in_features).
        self.W_Q = nn.Linear(in_features=qkv_input_dim, out_features=query_dim, bias=False)
        self.W_K = nn.Linear(in_features=qkv_input_dim, out_features=key_dim, bias=False)
        self.W_V = nn.Linear(in_features=qkv_input_dim, out_features=value_dim, bias=False)
        # 多个头连接后的线性变换.
        self.W_O = nn.Linear(in_features=value_dim, out_features=value_dim, bias=False)

    def forward(self, query_input, key_input, value_input, attn_mask=None, key_padding_mask=None):
        """
        前向传播方法.
        对于自注意力, query_input=X, key_input=X, value_input=X.

        :param key_padding_mask: 键填充掩码, shape=(batch_dim, kv_seq_dim).
        :param query_input: 查询的输入, shape=(batch_dim, q_seq_dim, n).
        :param key_input: 键的输入, shape: (batch_dim, kv_seq_dim, n).
        :param value_input: 值的输入, shape: (batch_dim, kv_seq_dim, n).
        :param attn_mask: 注意力掩码, shape=(kv_seq_dim, kv_seq_dim).
        :return: 多头注意力层输出.
        """
        # 计算h个头的查询、键和值.
        # query = (Q1,Q2,...,Qh), 计算query = query_input @ w^T, W^Q=w^T, 则Q=X * W^Q.
        # query: shape=(batch_dim, q_seq_dim, query_dim)
        query = self.W_Q(query_input)
        # key = (K1,K2,...,Kh), 计算K=X * W^K.
        # key: shape=(batch_dim, kv_seq_dim, key_dim).
        key = self.W_K(key_input)
        # value = (V1,V2,...,Vh), 计算V=X * W^V.
        # value: shape=(batch_dim, kv_seq_dim, value_dim).
        value = self.W_V(value_input)

        # 将h个头的查询、键和值分开到每个头上.
        # query : (batch_dim, num_heads, q_seq_dim, head_dim), query_dim = num_heads * head_dim.
        query = self.transpose_qkv(query)
        # key : (batch_dim, num_heads, kv_seq_dim, head_dim), key_dim = num_heads * head_dim.
        key = self.transpose_qkv(key)
        # value: (batch_dim, num_heads, kv_seq_dim, head_dim), value_dim = num_heads * head_dim.
        value = self.transpose_qkv(value)

        # 掩码变换.
        attn_mask = self.trans_mask(key_padding_mask=key_padding_mask,
                                    attn_mask=attn_mask, query=query)

        # 点击注意力, shape:(batch_dim, num_heads, kv_seq_dim, src_seq_len)
        h = self.dot_product_attention(query=query, key=key, value=value, attn_mask=attn_mask)

        # h: (batch_dim, q_seq_dim,num_heads,head_dim).
        h = h.transpose(1, 2)
        # cat_h: (batch_dim, q_seq_dim,head_dim * num_heads).
        cat_h = h.reshape(h.shape[0], h.shape[1], -1)  # h拼接.

        # attention: (batch_dim, q_seq_dim,value_dim).
        attention = self.W_O(cat_h)  # 多头注意力.

        return attention

    def transpose_qkv(self, qkv):
        """
        查询、键和值的输入维度变换.
        将shape:(batch_size,seq_len,qkv_dim)变为shape:(batch_size,seq_len,num_heads,head_dim).

        qkv_dim = num_heads * head_dim.

        :param qkv: shape:(batch_size,seq_len,qkv_dim).
        :return: qkv: shape:(batch_size,seq_len,num_heads,head_dim).
        """
        # 将qkv_dim分成两个维度: num_heads和head_dim.
        # qkv: (batch_size,seq_len,num_heads,head_dim).
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], self.num_heads, -1)

        # 将seq_len和head_dim两个维度放到最后.
        # qkv: shape=(batch_size,num_heads,seq_len,head_dim).
        qkv = torch.transpose(qkv, 1, 2)

        return qkv

    def trans_mask(self, key_padding_mask, attn_mask, query):
        """
        对于自注意力, q_seq_dim=kv_seq_dim.
        :param key_padding_mask: shape=(batch_dim, kv_seq_dim).
        :param attn_mask: shape=(kv_seq_dim, kv_seq_dim).
        :param query: (batch_dim, num_heads, q_seq_dim, head_dim).
        :return:
        """
        # 1. 增加头维度num_heads和查询序列维度q_seq_dim: (batch_dim, 1, 1, kv_seq_dim).
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        # 2. 沿着头扩展维度: (batch_dim, num_heads, 1, kv_seq_dim)
        key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
        # 3. 扩展到查询序列维度: (batch_dim, num_heads, q_seq_dim, kv_seq_dim)
        key_padding_mask = key_padding_mask.expand(-1, -1, query.size(-2), -1)

        if attn_mask is None:
            # attn_mask: (batch_dim, num_heads, q_seq_dim, kv_seq_dim).
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
        else:  # 扩展到批次和头维度.
            # attn_mask: shape=(1, 1, kv_seq_dim, kv_seq_dim)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            # attn_mask: shape=(1, num_heads, kv_seq_dim, kv_seq_dim)
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
            # attn_mask: shape=(batch_dim, num_heads, kv_seq_dim, kv_seq_dim)
            attn_mask = attn_mask.expand(query.size(0), -1, -1, -1)

        attn_mask = attn_mask + key_padding_mask

        return attn_mask


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout_p):
        super().__init__()
        # 多头注意力层.
        self.mah_layer = MultiHeadAttention(qkv_input_dim=embedding_dim, num_heads=num_heads,
                                            query_dim=embedding_dim, key_dim=embedding_dim, value_dim=embedding_dim)
        self.mah_dropout = Dropout(dropout_p)
        self.mah_add_norm = AddNorm(embedding_dim)  # 多头注意力层的层归一化.

    def forward(self, query_input, key_input, value_input, attn_mask=None, key_padding_mask=None):
        """

        :param key_padding_mask: shape=(batch_dim, kv_seq_dim).
        :param attn_mask: shape=(kv_seq_dim, kv_seq_dim).
        :param query_input: 查询输入, shape:(batch_dim, q_seq_dim, query_dim), query_dim=d_k.
        :param key_input: 键输入, shape:(batch_dim, kv_seq_dim, key_dim), key_dim=d_k.
        :param value_input: 值输入, shape:(batch_dim, kv_seq_dim, value_dim), value_dim=d_v.
        :return: 输出: shape:(batch_dim, q_seq_dim, n).
        """
        # mah_out: shape=(batch_dim, q_seq_dim, value_dim).
        mah_out = self.mah_layer(query_input, key_input, value_input, attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask)  # 多头注意力.
        mah_out = self.mah_dropout(mah_out)  # dropout.
        mah_out = self.mah_add_norm(query_input, mah_out)  # 标准化.

        return mah_out


class PositionWiseFFN(nn.Module):
    """
    基于位置的前馈神经网络层.
    """

    def __init__(self, num_input=512, num_hiddens=2048, num_outputs=512, dropout_p=0.1):
        """
        初始化方法.
        :param num_input: FFN的输入维度.
        :param num_hiddens: 隐藏层神经元个数.
        :param num_outputs: FFN输出维度.
        :param dropout_p: Dropout概率.
        """
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(num_input, num_hiddens)  # 第一个线性.
        self.activation = nn.ReLU()  # 激活函数.
        self.activation_dropout = Dropout(dropout_p)  # 激活后的Dropout.
        self.W_2 = nn.Linear(num_hiddens, num_outputs)  # 第二个线性.

    def forward(self, X):
        """

        :param X: shape: (batch_dim, q_seq_len, num_input).
        :return: shape: (batch_dim, q_seq_len, num_outputs).
        """
        out = self.W_1(X)
        out = self.activation(out)
        out = self.activation_dropout(out)
        out = self.W_2(out)

        return out


class PositionWiseFFNBlock(nn.Module):
    """
    基于位置的前馈神经网络层.
    """

    def __init__(self, num_input=512, num_hiddens=2048, num_outputs=512, dropout_p=0.1):
        """
        初始化方法.
        :param num_input: FFN的输入维度.
        :param num_hiddens: 隐藏层神经元个数.
        :param num_outputs: FFN输出维度.
        :param dropout_p: Dropout概率.
        """
        super().__init__()
        # 前馈神经网络层.
        self.ffn = PositionWiseFFN(num_input=num_input, num_hiddens=num_hiddens,
                                   num_outputs=num_outputs, dropout_p=dropout_p)  # 前馈神经网络层.
        self.ffn_dropout = Dropout(dropout_p)  # 前馈神经网层dropout.
        self.ffn_add_norm = AddNorm(num_outputs)  # 前馈神经网络层归一化.

    def forward(self, X):
        """

        :param X: shape: (batch_dim, q_seq_len, num_input).
        :return: shape: (batch_dim, q_seq_len, num_outputs).
        """
        ffn_block_out = self.ffn(X)  # 前馈神经网络.
        ffn_block_out = self.ffn_dropout(ffn_block_out)  # dropout.
        ffn_block_out = self.ffn_add_norm(X, ffn_block_out)  # 残差连接和标准化.

        return ffn_block_out


class TransformerEncoderLayer(nn.Module):
    """
    编码器层.
    它的输出将作为解码器中每个decoder层多头注意力的key_input和value_input.
    """

    def __init__(self, num_heads, embedding_dim, feedforward_dim, dropout_p):
        """
        初始化方法.
        :param num_heads: 头的数量.
        :param embedding_dim: embedding维度.
        :param feedforward_dim: 前馈神经网络层隐藏层维度大小.
        :param dropout_p: dropout概率.
        """

        super().__init__()
        # 多头注意力层.
        self.mah_block = MultiHeadAttentionBlock(num_heads, embedding_dim, dropout_p)

        # 前馈神经网络层.
        self.ffn_block = PositionWiseFFNBlock(num_input=embedding_dim, num_hiddens=feedforward_dim,
                                              num_outputs=embedding_dim, dropout_p=dropout_p)  # 前馈神经网络层.

    def forward(self, query_input, key_input, value_input, src_mask=None, src_key_padding_mask=None):
        """

        :param src_key_padding_mask: shape=(batch_dim, kv_seq_dim).
        :param src_mask: shape=(kv_seq_dim, kv_seq_dim).
        :param query_input: 查询输入, shape:(batch_dim, q_seq_dim, query_dim), query_dim=d_k.
        :param key_input: 键输入, shape:(batch_dim, kv_seq_dim, key_dim), key_dim=d_k.
        :param value_input: 值输入, shape:(batch_dim, kv_seq_dim, value_dim), value_dim=d_v.
        :return: 输出: shape:(batch_dim, q_seq_dim, n).
        """
        # mah_out: shape=(batch_dim, q_seq_dim, value_dim).
        mah_out = self.mah_block(query_input, key_input, value_input, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)  # 多头注意力.
        ffn_block_out = self.ffn_block(mah_out)  # 前馈神经网络.

        return ffn_block_out


class TransformerDecoderLayer(nn.Module):
    """
    解码器层.
    """

    def __init__(self, num_heads, embedding_dim, feedforward_dim, dropout_p):
        super().__init__()
        # 掩码多头注意力层.
        self.masked_mah_block = MultiHeadAttentionBlock(num_heads, embedding_dim, dropout_p)

        # 编码器-解码器注意力层, 本质上就是编码层, 只是输入不同.
        self.encoder_decoder_mha_layer = TransformerEncoderLayer(num_heads, embedding_dim, feedforward_dim, dropout_p)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """

        :param tgt: 目标序列, shape=(batch_dim, tgt_seq_dim, n).
        :param memory: 编码器输出, shape=(batch_dim, src_q_seq_dim, n).
        :param tgt_mask: 目标序列未来掩码, shape=(tgt_kv_seq_dim, tgt_kv_deq_dim).
        :param tgt_key_padding_mask: 目标序列键填充掩码, shape=(batch_dim, tgt_kv_seq_dim).
        :param memory_key_padding_mask: 内存键填充掩码, shape=(batch_dim, src_kv_seq_dim).
        :return:
        """

        # 掩码注意力层输出.
        mml_out = self.masked_mah_block(query_input=tgt, key_input=tgt, value_input=tgt, attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask)

        # 编码器-解码器注意力层和神经网络层输出.
        decoder_out = self.encoder_decoder_mha_layer(query_input=mml_out, key_input=memory, value_input=memory,
                                                     src_key_padding_mask=memory_key_padding_mask)

        return decoder_out


class TransformerEncoder(nn.Module):
    """
    编码器: 堆叠了N个编码器层.
    """

    def __init__(self, num_encoder_layers, num_heads, embedding_dim, feedforward_dim, dropout_p):
        """
        初始化方法.
        :param num_encoder_layers: 编码器层的数量.
        :param embedding_dim: embedding维度.
        :param feedforward_dim 前馈神经网络隐藏层维度.
        :param dropout_p dropout概率.
        """
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        # 作为TransformerEncoder的子模块, 使用[]再调用to()方法无法统一迁移至device上.
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_encoder_layers):  # 初始化每个编码层.
            self.tel = TransformerEncoderLayer(embedding_dim=embedding_dim, num_heads=num_heads,
                                               feedforward_dim=feedforward_dim, dropout_p=dropout_p)
            self.encoder_layers.append(self.tel)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """

        :param src_key_padding_mask: 源序列键填充掩码.
        :param src_mask: 原序列注意力掩码.
        :param src: shape: (batch_size,seq_len,embedding_size)
        :return: 编码器输出.
        """
        for i in range(self.num_encoder_layers):
            encoder_layer = self.encoder_layers[i]  # 第i个编码器层.
            src = encoder_layer(query_input=src, key_input=src, value_input=src,
                                src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, num_decoder_layers, num_heads, embedding_dim, feedforward_dim, dropout_p):
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        # 作为TransformerDecoder的子模块, 使用[]再调用to()方法无法统一迁移至device上.
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_decoder_layers):  # 初始化每个解码层.
            self.tdl = TransformerDecoderLayer(embedding_dim=embedding_dim, num_heads=num_heads,
                                               feedforward_dim=feedforward_dim, dropout_p=dropout_p)
            self.decoder_layers.append(self.tdl)

    def forward(self, tgt, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """

        :param tgt_key_padding_mask:
        :param memory_key_padding_mask:
        :param tgt_mask:
        :param memory: encoder输出,作为decoder的key和value.
        :param tgt: shape: (batch_size,seq_len,embedding_size).
        :return: 编码器输出.
        """
        out = tgt
        for i in range(self.num_decoder_layers):
            decoder_layer = self.decoder_layers[i]  # 第i个编码器层.
            out = decoder_layer(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)

        return out


class UserTransformer(nn.Module):
    """
    Transformer模型.
    """

    def __init__(self, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 feedforward_dim=2048, dropout_p=0.1):
        super().__init__()

        # 编码器.
        self.encoder = TransformerEncoder(num_encoder_layers=num_encoder_layers, num_heads=num_heads,
                                          embedding_dim=d_model, feedforward_dim=feedforward_dim,
                                          dropout_p=dropout_p)
        # 解码器.
        self.decoder = TransformerDecoder(num_decoder_layers=num_decoder_layers, num_heads=num_heads,
                                          embedding_dim=d_model, feedforward_dim=feedforward_dim,
                                          dropout_p=dropout_p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """

        :param memory_key_padding_mask: 编码器-解码器层键填充掩码.
        :param tgt_key_padding_mask:  目标键的填充掩码, 用于QK^T/sqrt(d_k)之后,softmax之前.
        :param tgt_mask:  目标序列的注意力掩码.
        :param src_mask: 源序列的注意力掩码.
        :param src_key_padding_mask: 源序列键填充掩码, 用于QK^T/sqrt(d_k)之后,softmax之前.
        :param src:  shape:(batch_size,seq_len).
        :param tgt: shape:(batch_size,seq_len,embedding_size).
        :return: 概率: shape: (batch_size,seq_len,voc_size), voc_size: 字典大小.
        """
        # 编码.
        memory = self.encoder(src=src, src_mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        # 解码. shape(batch_size,seq_len,embedding_dim).
        decoder_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)

        return decoder_out


class TransformerModel(nn.Module):
    def __init__(self, num_embeddings, pad_id, config, transformer_type='pytorch'):
        super(TransformerModel, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = config.embedding_dim
        self.dropout_p = config.dropout_p
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_head
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.feedforward_dim = config.feedforward_dim
        self.pad_id = pad_id
        self.device = config.device
        self.best_model_name = 'best_translation_model.pt'
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

        self.mask = Masker(padding_id=pad_id, device=self.device)  # 掩码器.
        self.embedding = Embedding(num_embeddings, self.embedding_dim)  # 共享嵌入层.
        self.positional_encoding = PositionalEncoding(
            pe_dim=self.embedding_dim, dropout_p=self.dropout_p, max_seq_len=self.max_seq_len)  # 位置编码.
        if transformer_type == 'pytorch':
            # 内置transformer: 仅包含编码器和解码器.
            self.transformer = nn.Transformer(
                d_model=self.embedding_dim, nhead=self.num_heads,
                num_encoder_layers=self.num_encoder_layers, num_decoder_layers=self.num_decoder_layers,
                dim_feedforward=self.feedforward_dim, dropout=self.dropout_p,
                bias=False, batch_first=True  # 使用(batch, seq, feature)格式.
            )
        elif transformer_type == 'user':
            self.transformer = UserTransformer(
                d_model=self.embedding_dim, num_heads=self.num_heads,
                num_encoder_layers=self.num_encoder_layers, num_decoder_layers=self.num_decoder_layers,
                feedforward_dim=self.feedforward_dim, dropout_p=self.dropout_p,
            )
        else:
            raise Exception(f"不支持{transformer_type}")
        # 输出线性层.
        self.fc_out = nn.Linear(self.embedding_dim, num_embeddings)

    def forward(self, src_input: Tensor, tgt_input: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None, src_key_padding_mask: Tensor = None,
                tgt_key_padding_mask: Tensor = None) -> Tensor:
        """
        前向传播.
        :param src_input: 源输入序列.
        :param tgt_input: 目标输入序列.
        :param src_mask: 源序列注意力掩码.
        :param tgt_mask: 目标序列注意力掩码.
        :param src_key_padding_mask: 源序列键填充掩码.
        :param tgt_key_padding_mask: 目标序列键填充掩码.
        :return:
        """
        # 嵌入层 + 位置编码
        src_pe_out = self.before(src_input)
        tgt_pe_out = self.before(tgt_input)

        # Transformer前向传播, 编码器-解码器层注意力使用了键填充掩码memory_key_padding_mask.
        outs = self.transformer(
            src=src_pe_out, tgt=tgt_pe_out,
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return self.fc_out(outs)

    def before(self, inputs):
        """
        transformer前置操作:embedding + positional encoding.
        :param inputs:
        :return:
        """
        out = self.embedding(inputs)
        out = out * math.sqrt(self.embedding_dim)
        out = self.positional_encoding(out)

        return out

    def encode(self, src_input: Tensor, src_key_padding_mask: Tensor = None):
        """
        编码.
        :param src_input: 编码器输入.
        :param src_key_padding_mask: 源序列键填充掩码.
        :return: 编码器输出.
        """
        src_pe_out = self.before(src_input)
        return self.transformer.encoder(src_pe_out, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt_input: Tensor, memory: Tensor, tgt_mask: Tensor, memory_key_padding_mask: Tensor,
               tgt_key_padding_mask):
        """
        解码.
        :param tgt_input: 目标序列输出.
        :param memory: 编码器输出.
        :param tgt_mask: 目标序列注意力掩码.
        :param memory_key_padding_mask: 编码器-解码器层的键填充掩码.
        :param tgt_key_padding_mask: 目标序列键填充掩码.
        :return:
        """
        tgt_pe_out = self.before(tgt_input)

        return self.transformer.decoder(
            tgt_pe_out, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)

    def compute_batch_loss(self, src_input, tgt, optimizer, loss_fn):
        """
        计算一个批次损失.
        :param src_input: 一个源序列批次输入.
        :param tgt: 一个目标序列批次输入.
        :param optimizer: 优化器.
        :param loss_fn: 损失函数.
        :return:
        """
        # 创建decoder的输入和输出.
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]  # 排除序列开始的<bos>.

        # 创建掩码.
        src_padding_mask = self.mask.create_key_padding_mask(src_input)
        tgt_padding_mask = self.mask.create_key_padding_mask(tgt_input)
        tgt_mask = self.mask.create_future_mask(tgt_input)

        # 模型输出 shape:(batch_dim, seq_dim, vocab_dim).
        logits = self.forward(
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

    def fit(self, epochs, train_loader, val_loader, optimizer, loss_fn):
        """

        :param loss_fn:
        :param train_loader:
        :param val_loader:
        :param optimizer:
        :param pad_id:
        :return:
        """
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # 平均批次损失.
            train_loss = self.train_epoch(train_loader, optimizer, loss_fn)
            val_loss = self.evaluate(val_loader, optimizer, loss_fn)
            self.epochs.append(epoch + 1)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss: .4f} | Val Loss: {val_loss: .4f}')
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'epoch': epoch, 'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'loss': val_loss,
                            }, self.best_model_name)
        print(f"训练完成！模型已保存为 '{self.best_model_name}'")

    def train_epoch(self, train_loader, optimizer, loss_fn):
        """
        训练一轮.
        :param train_loader: 训练数据.
        :param optimizer: 优化器
        :param loss_fn: 损失函数.
        :param pad_id:
        :return:
        """
        self.train()  # 开启模型训练模式.
        total_loss = 0  # 所有批次的总损失.
        batch_interval_loss = 0  # 批次区间损失.
        batch_num = 1  # 批次号.
        for src_ids, tgt_ids, _, _ in train_loader:  # 遍历批次.
            optimizer.zero_grad()
            batch_loss = self.compute_batch_loss(src_ids, tgt_ids, optimizer, loss_fn)  # 一个批次.
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()  # 累积每批损失.
            batch_interval_loss += batch_loss.item()
            batch_num += 1
        average_batch_loss = total_loss / len(train_loader)  # 平均每批损失（该批次中所有样本的损失）.

        return average_batch_loss

    # 评估函数
    def evaluate(self, val_loader, optimizer, loss_fn):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for src_input, tgt, _, _ in val_loader:
                loss = self.compute_batch_loss(src_input, tgt, optimizer, loss_fn)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def predict(self, src_input, key_token_ids):
        self.eval()
        # 生成源序列填充掩码.
        src_padding_mask = self.mask.create_key_padding_mask(src_input)
        # 进行预测
        _, bos_id, eos_id, _ = key_token_ids
        # 编码器输出
        memory = self.encode(src_input, src_padding_mask)
        # 初始化解码器输入 (起始符号) <bos>.
        outputs = torch.ones(src_input.size(0), 1).fill_(bos_id).type_as(src_input).to(self.device)
        eos_flags = torch.zeros(outputs.size(0), dtype=torch.bool).to(self.device)  # 终止标志

        for i in range(self.max_seq_len - 1):
            tgt_mask = self.mask.create_future_mask(outputs).type_as(src_padding_mask).to(self.device)
            tgt_padding_mask = self.mask.create_key_padding_mask(outputs)
            # 解码器输出
            out = self.decode(outputs, memory, tgt_mask, memory_key_padding_mask=src_padding_mask,
                              tgt_key_padding_mask=tgt_padding_mask)
            # 预测下一个token
            prob = self.fc_out(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)
            # 添加到序列
            outputs = torch.cat([outputs, next_word.unsqueeze(1)], dim=1)
            # 如果遇到结束符则停止
            # 更新终止标志
            eos_flags |= (next_word == eos_id)
            # 所有序列遇到EOS则终止
            if eos_flags.all():
                break

        return outputs

    def load(self):
        print("加载模型...")
        checkpoint = torch.load(self.best_model_name)
        self.load_state_dict(checkpoint['model_state_dict'])
        print('模型加载成功')
