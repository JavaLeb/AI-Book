from torch import nn
import torch
import math


class Embedding(nn.Module):
    """
    嵌入层(直接使用内置实现).
    """

    def __init__(self, num_embeddings, embedding_dim):
        """
        初始化方法.
        :param num_embeddings: embedding的token数量.
        :param embedding_dim: embedding向量维度.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, X):
        return self.embedding(X)


class PositionalEncoding(nn.Module):
    r"""位置编码层，获取序列中token之间的相对或绝对位置信息.
        位置编码向量与embedding向量维度相同, 因此它们可以求和.
        使用不同频率的sine和cosine函数.
    """

    def __init__(self, embedding_dim, dropout_p=0.1, max_len=5000):
        """

        :param embedding_dim: embedding向量的维度大小.
        :param dropout_p: dropout值，默认0.1.
        :param max_len: 输入序列最大长度.
        """
        super(PositionalEncoding, self).__init__()  # 父类初始化.
        self.dropout = nn.Dropout(p=dropout_p)  # 初始化Dropout.

        pe = torch.zeros(max_len, embedding_dim)  # 行表示序列长度, 列表示位置向量维度, shape: (seq_len,embedding_size).
        # unsqueeze在指定维度dim插入长度为1的新维度,例如shape=(3,)，当dim=1，则变为shape=(3,1).
        # (seq_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)  # 生成足够长的位置pos.
        # 使用exp避免精度丢失, 相当于公式中的w_2i.
        w_2i = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(w_2i * position)  # 偶数位置使用正弦函数.
        pe[:, 1::2] = torch.cos(w_2i * position)  # 奇数位置使用余弦函数.
        # 初始pe的形状(seq_len,embedding_size), 然后在dim=0插入批次维度变为(1,seq_len,d_model).
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 注册, 调用to()方法时, 可迁移计算device.

    def forward(self, X):
        r"""
        输出使用dropout.
        :param X: 输入 [batch_size, seq_len, embed dim].
        形状:
            输入: [batch_size, seq_len, embed dim]
            输出: [batch_size, seq_len, embed dim]
        """
        # x.size(1)获取序列长度, self.pe[:,:x.size(1), :]的shape(1,35,512), 通过在批次维度上广播与x求和.
        X = X + self.pe[:, :X.size(1), :]

        return self.dropout(X)  # 以某个概率将x的值变为0.


class LayerNorm(nn.Module):
    """
    层标准化, 默认按照最后一个维度标准化.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, X):
        return self.layer_norm(X)


class MultiAttentionHead(nn.Module):
    def __init__(self, qkv_input_dim, num_heads, query_dim, key_dim, value_dim, attn_mask=None) -> None:
        """

        :param qkv_input_dim: embedding_size
        :param num_heads: 头的数量.
        :param query_dim: 查询的维度d_k.
        :param key_dim: 键的维度d_k.
        :param value_dim: 值的维度d_v.
        """
        super().__init__()

        self.qkv_input_dim = qkv_input_dim  # n=d_model.
        self.num_heads = num_heads  # h.
        self.query_dim = query_dim  # d_k.
        self.key_dim = key_dim  # d_k.
        self.value_dim = value_dim  # d_v.

        self.num_heads = num_heads
        self.head_dim = qkv_input_dim // num_heads  # 需要整除.
        # 线性模型权重w的shape(out_features,in_features).
        self.W_Q = nn.Linear(in_features=qkv_input_dim, out_features=query_dim, bias=False)
        self.W_K = nn.Linear(in_features=qkv_input_dim, out_features=key_dim, bias=False)
        self.W_V = nn.Linear(in_features=qkv_input_dim, out_features=value_dim, bias=False)

        self.W_O = nn.Linear(value_dim, value_dim, bias=False)

    def forward(self, query_input, key_input, value_input, tgt_mask=None):
        """
        对于自注意力, query_input=X, key_input=X, value_input=X.
        :param tgt_mask: 目标序列掩码.
        :param query_input: 查询输入, shape: (batch_size,seq_len,n), n=d_model.
        :param key_input: 键输入, shape: (batch_size,seq_len,n).
        :param value_input: 值输入, shape: (batch_size,seq_len,n).
        :return: 多头注意力层输出.
        """

        # query = (Q1,Q2,...,Qh).
        query = self.W_Q(query_input)  # 相当于计算  query = query_input @ w^T, w^T看成WQ, 则Q=X * WQ.
        # key = (K1,K2,...,Kh).
        key = self.W_K(key_input)  # K=X * WK.
        # value = (V1,V2,...,Vh).
        value = self.W_V(value_input)  # V=X * WV.

        query = self.transpose_qkv(query)
        key = self.transpose_qkv(key)
        value = self.transpose_qkv(value)

        # query : (batch_size,num_heads,seq_len,head_dim)
        # key : (batch_size,num_heads,seq_len,head_dim)
        # qk : (batch_size,num_heads,seq_len,seq_len)
        qk = query @ torch.transpose(key, -1, -2)
        scaled_qk = qk / math.sqrt(self.query_dim)
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        h = softmax_qk @ value
        cat_h = h.reshape(h.shape[0], h.shape[2], -1)  # h拼接.

        attention = self.W_O(cat_h)  # 多头注意力.

        return attention

    def transpose_qkv(self, qkv):
        """
        head_dim = qkv_dim // num_heads.

        :param qkv: shape:(batch_size,seq_len,qkv_dim).
        :return: qkv: shape:(batch_size,seq_len,num_heads,head_dim).
        """
        # 将qkv_dim分成两个维度: num_heads和head_dim.
        # shape:(batch_size,seq_len,num_heads,head_dim).
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], self.num_heads, -1)

        # 将seq_len和head_dim两个维度放到最后.
        # shape:(batch_size,num_heads,seq_len,head_dim).
        qkv = torch.transpose(qkv, 1, 2)

        return qkv


class PositionWiseFFN(nn.Module):
    """
    基于位置的前馈神经网络层.
    """

    def __init__(self, num_input, num_hiddens, num_outputs, dropout_p):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(num_input, num_hiddens)
        self.activation = nn.ReLU()
        self.activation_dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        out = self.activation(self.linear1(X))
        out = self.activation_dropout(out)
        out = self.linear2(out)

        return out


class CommonBlock(nn.Module):
    """
    encoder和decoder的公共模块, 包括多头注意力层和前馈神经网络层, 以及对应的Add & Norm.
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
        self.mah_layer = MultiAttentionHead(qkv_input_dim=embedding_dim, num_heads=num_heads,
                                            query_dim=embedding_dim, key_dim=embedding_dim, value_dim=embedding_dim)
        self.mah_dropout = nn.Dropout(p=dropout_p)
        self.mah_norm = LayerNorm(embedding_dim)  # 多头注意力层的层归一化.

        # 前馈神经网络层.
        self.ffn = PositionWiseFFN(num_input=embedding_dim, num_hiddens=feedforward_dim,
                                   num_outputs=embedding_dim, dropout_p=dropout_p)  # 前馈神经网络层.
        self.ffn_dropout = nn.Dropout(p=dropout_p)  # 前馈神经网层dropout.
        self.ffn_norm = LayerNorm(embedding_dim)  # 前馈神经网络层归一化.

    def forward(self, query_input, key_input, value_input):
        """

        :param value_input: 值输入, shape:(batch_size,seq_len,value_dim), value_dim=d_v.
        :param key_input: 键输入, shape:(batch_size,seq_len,key_dim), key_dim=d_k.
        :param query_input: 查询输入, shape:(batch_size,seq_len,query_dim), query_dim=d_k.
        :return: 输出: shape:(batch_size,seq_len,value_dim).
        """
        mah_out = self.mah_layer(query_input, key_input, value_input)  # 多头注意力.
        mah_out = self.mah_dropout(mah_out)  # dropout.
        ffn_input = self.mah_norm(query_input + mah_out)  # 残差连接和标准化.

        ffn_out = self.ffn(ffn_input)  # 前馈神经网络.
        ffn_out = self.ffn_dropout(ffn_out)  # dropout.
        ffn_out = self.ffn_norm(ffn_input + ffn_out)  # 残差连接和标准化.

        return ffn_out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim, feedforward_dim, dropout_p):
        super().__init__()
        # 掩码多头注意力层.
        self.masked_mah_layer = MultiAttentionHead(qkv_input_dim=embedding_dim, num_heads=num_heads,
                                                   query_dim=embedding_dim, key_dim=embedding_dim,
                                                   value_dim=embedding_dim)
        self.masked_mah_dropout = nn.Dropout(p=dropout_p)
        self.masked_mah_norm = LayerNorm(embedding_dim=embedding_dim)

        # 初始化: 多头注意层和前馈神经网络层.
        self.common_block = CommonBlock(num_heads, embedding_dim, feedforward_dim, dropout_p)

    def forward(self, encoder_out, tgt, tgt_mask=None):
        # 掩码注意力层输出.
        mml_out = self.masked_mah_layer(query_input=tgt, key_input=tgt, value_input=tgt, tgt_mask=tgt_mask)
        mml_out = self.masked_mah_dropout(mml_out)
        mml_out = self.masked_mah_norm(tgt + mml_out)  # 残差连接和标准化.

        # 注意力层和神经网络层输出.
        decoder_out = self.common_block(query_input=mml_out, key_input=encoder_out, value_input=encoder_out)

        return decoder_out


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

    def forward(self, encoder_out, Y):
        """

        :param encoder_out: encoder输出,作为decoder的key和value.
        :param Y: shape: (batch_size,seq_len,embedding_size).
        :return: 编码器输出.
        """

        for i in range(self.num_decoder_layers):
            decoder_layer = self.decoder_layers[i]  # 第i个编码器层.
            Y = decoder_layer(encoder_out, Y)

        return Y


class TransformerEncoderLayer(CommonBlock):
    """
    编码器层: 与公共层一致, 直接继承即可.

    它的输出将作为解码器中每个decoder层多头注意力的key_input和value_input.
    """


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

    def forward(self, X):
        """

        :param X: shape: (batch_size,seq_len,embedding_size)
        :return: 编码器输出.
        """
        for i in range(self.num_encoder_layers):
            encoder_layer = self.encoder_layers[i]  # 第i个编码器层.
            X = encoder_layer(query_input=X, key_input=X, value_input=X)

        return X


class Transformer(nn.Module):
    """
    Transformer模型.
    """

    def __init__(self, num_embeddings, embedding_dim, feedforward_dim, dropout_p,
                 num_heads, num_encoder_layers, num_decoder_layers):
        super().__init__()

        # embedding层. 权重w的shape(num_embeddings,embedding_dim)
        self.enc_embedding_layer = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 位置编码层.
        self.enc_pe_layer = PositionalEncoding(embedding_dim=embedding_dim, dropout_p=dropout_p)
        # Transformer模型编码器.
        self.encoder = TransformerEncoder(num_encoder_layers=num_encoder_layers, num_heads=num_heads,
                                          embedding_dim=embedding_dim, feedforward_dim=feedforward_dim,
                                          dropout_p=dropout_p)

        self.dec_embedding_layer = self.enc_embedding_layer  # 共享embedding.
        self.dec_pe_layer = self.enc_pe_layer  # 编码器和解码器位置编码相同(因为位置编码仅与位置有关).
        self.decoder = TransformerDecoder(num_decoder_layers=num_decoder_layers, num_heads=num_heads,
                                          embedding_dim=embedding_dim, feedforward_dim=feedforward_dim,
                                          dropout_p=dropout_p)
        # shape: (num_embeddings,embedding_dim).
        self.decoder_linear_weight = self.enc_embedding_layer.embedding.weight  # 权重共享.

    def forward(self, src_input, target_input):
        """

        :param src_input:  shape:(batch_size,seq_len).
        :param target_input: shape:(batch_size,seq_len,embedding_size).
        :return: 概率: shape: (batch_size,seq_len,voc_size), voc_size: 字典大小.
        """
        # 对应transformer模型架构图左侧部分.
        # embedding结果输出shape(batch_size,seq_len,embedding_size)
        encoder_embedding_out = self.enc_embedding_layer(src_input)  # embedding层输出.
        encoder_pe_out = self.enc_pe_layer(encoder_embedding_out)  # 位置编码层输出.
        encoder_out = self.encoder(encoder_pe_out)  # transformer.

        # 对应transformer模型架构图右侧部分.
        decoder_embedding_out = self.dec_embedding_layer(target_input)  # decoder embedding层.
        decoder_pe_out = self.dec_pe_layer(decoder_embedding_out)  # 位置编码层.
        decoder_out = self.decoder(encoder_out, decoder_pe_out)  # shape(batch_size,seq_len,embedding_dim).
        # 结果shape (batch_size,seq_len,num_embeddings).
        out = decoder_out @ self.decoder_linear_weight.t()  # 未经过softmax归一化的logits值.

        # out = torch.softmax(out, dim=-1)  # decoder的softmax层.

        return out
