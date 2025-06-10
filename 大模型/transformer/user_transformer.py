from torch import nn
import torch
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
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, X):
        """
        根据索引查找词嵌入.
        :param X:  X的值表示token的索引, shape=(batch_size, seq_len).
        :return:
        """
        return self.embedding(X)


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


class DotProductAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, key, query, value, attn_mask):
        # qk : (batch_size,num_heads,src_seq_len,tgt_seq_len).
        qk = query @ torch.transpose(key, -1, -2)  # 可使用torch.bmm()批量矩阵乘法.
        # scaled_qk: (batch_size,num_heads,src_seq_len,tgt_seq_len).
        scaled_qk = qk / math.sqrt(self.head_dim)
        scaled_qk += attn_mask
        # softmax_qk: (batch_size,num_heads,src_seq_len,tgt_seq_len).
        attention_weights = torch.softmax(scaled_qk, dim=-1)
        # h: (batch_size,num_heads,src_seq_len,head_dim).
        # 有些扩展在权重上使用了Dropout.
        h = attention_weights @ value

        return h


class MultiAttentionHead(nn.Module):
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

        self.dot_product_attention = DotProductAttention(self.head_dim)

        # 查询、键和值的权重初始化. 线性模型权重w的shape=(out_features,in_features).
        self.W_Q = nn.Linear(in_features=qkv_input_dim, out_features=query_dim, bias=False)
        self.W_K = nn.Linear(in_features=qkv_input_dim, out_features=key_dim, bias=False)
        self.W_V = nn.Linear(in_features=qkv_input_dim, out_features=value_dim, bias=False)
        # 多个头连接后的线性变换.
        self.W_O = nn.Linear(in_features=value_dim, out_features=value_dim, bias=False)

    def forward(self, query_input, key_input, value_input, attn_mask=None, key_padding_mask=None):
        """
        对于自注意力, query_input=X, key_input=X, value_input=X.

        :param key_padding_mask: 键填充掩码.
        :param query_input: 查询的输入, shape=(batch_size,src_seq_len,n), n=d_model.
        :param key_input: 键的输入, shape: (batch_size,tgt_seq_len,n).
        :param value_input: 值的输入, shape: (batch_size,tgt_seq_len,n).
        :param attn_mask: 注意力掩码.
        :return: 多头注意力层输出.
        """

        # query = (Q1,Q2,...,Qh), 计算query = query_input @ w^T, WQ=w^T, 则Q=X * WQ.
        # query: shape=(batch_size,src_seq_len,query_dim)
        query = self.W_Q(query_input)
        # key = (K1,K2,...,Kh), 计算K=X * WK.
        # key: shape=(batch_size,tgt_seq_len,key_dim).
        key = self.W_K(key_input)
        # value = (V1,V2,...,Vh), 计算V=X * WV.
        # value: shape=(batch_size,tgt_seq_len,value_dim).
        value = self.W_V(value_input)

        # query : (batch_size,num_heads,src_seq_len,head_dim), query_dim=num_heads*head_dim.
        query = self.transpose_qkv(query)
        # key : (batch_size,num_heads,tgt_seq_len,head_dim), key_dim=num_heads*head_dim.
        key = self.transpose_qkv(key)
        # value: (batch_size,num_heads,tgt_seq_len,head_dim), value_dim=num_heads*head_dim.
        value = self.transpose_qkv(value)

        # 点击注意力, shape:(batch_size,num_heads,tgt_seq_len,src_seq_len)
        # 1. 增加维度: (batch_size, 1, 1, tgt_seq_len)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        # 2. 扩展维度: (batch_size, num_heads, 1, tgt_seq_len)
        key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)
        # 3. 扩展到源序列长度: (batch_size, num_heads, tgt_seq_len, src_seq_len)
        key_padding_mask = key_padding_mask.expand(-1, -1, -1, key.size(-2))

        if attn_mask is None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)  # 注意力偏置项.
        attn_mask = self.trans_mask(key_padding_mask, attn_mask)
        h = self.dot_product_attention(query=query, key=key, value=value, attn_mask=attn_mask)

        # h: (batch_size,src_seq_len,num_heads,head_dim).
        h = h.transpose(1, 2)
        # cat_h: (batch_size,src_seq_len,head_dim * num_heads).
        cat_h = h.reshape(h.shape[0], h.shape[1], -1)  # h拼接.

        # attention: (batch_size,src_seq_len,value_dim).
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

    def trans_mask(self, key_padding_mask, attn_mask):
        # 叠加填充掩码和注意力掩码, 填充掩码如果是bool型, 事先转换成0/-inf.
        if key_padding_mask is not None:
            _mask_dtype = key_padding_mask.dtype
            _mask_is_float = torch.is_floating_point(key_padding_mask)
            if _mask_dtype != torch.bool and not _mask_is_float:
                raise AssertionError(
                    f"only bool and floating types of attention are supported"
                )
            if not _mask_is_float:
                key_padding_mask = torch.zeros_like(key_padding_mask, dtype=torch.float).masked_fill_(
                    key_padding_mask, float("-inf")
                )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask += key_padding_mask

        return attn_mask


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
        self.activation_dropout = nn.Dropout(dropout_p)  # 激活后的Dropout.
        self.W_2 = nn.Linear(num_hiddens, num_outputs)  # 第二个线性.

    def forward(self, X):
        """

        :param X:
        :return:
        """
        out = self.W_1(X)
        out = self.activation(out)
        out = self.activation_dropout(out)
        out = self.W_2(out)

        return out


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
        self.mah_layer = MultiAttentionHead(qkv_input_dim=embedding_dim, num_heads=num_heads,
                                            query_dim=embedding_dim, key_dim=embedding_dim, value_dim=embedding_dim)
        self.mah_dropout = nn.Dropout(p=dropout_p)
        self.mah_add_norm = AddNorm(embedding_dim)  # 多头注意力层的层归一化.

        # 前馈神经网络层.
        self.ffn = PositionWiseFFN(num_input=embedding_dim, num_hiddens=feedforward_dim,
                                   num_outputs=embedding_dim, dropout_p=dropout_p)  # 前馈神经网络层.
        self.ffn_dropout = nn.Dropout(p=dropout_p)  # 前馈神经网层dropout.
        self.ffn_add_norm = AddNorm(embedding_dim)  # 前馈神经网络层归一化.

    def forward(self, query_input, key_input, value_input, src_mask=None, src_key_padding_mask=None):
        """

        :param query_input: 查询输入, shape:(batch_size,src_seq_len,query_dim), query_dim=d_k.
        :param key_input: 键输入, shape:(batch_size,tgt_seq_len,key_dim), key_dim=d_k.
        :param value_input: 值输入, shape:(batch_size,tgt_seq_len,value_dim), value_dim=d_v.
        :return: 输出: shape:(batch_size,seq_len,value_dim).
        """
        # (batch_size,src_seq_len,value_dim).
        mah_out = self.mah_layer(query_input, key_input, value_input, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)  # 多头注意力.
        mah_out = self.mah_dropout(mah_out)  # dropout.
        ffn_input = self.mah_add_norm(query_input, mah_out)  # 标准化.

        ffn_out = self.ffn(ffn_input)  # 前馈神经网络.
        ffn_out = self.ffn_dropout(ffn_out)  # dropout.
        ffn_out = self.ffn_add_norm(ffn_input, ffn_out)  # 残差连接和标准化.

        return ffn_out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim, feedforward_dim, dropout_p):
        super().__init__()
        # 掩码多头注意力层.
        self.masked_mah_layer = MultiAttentionHead(qkv_input_dim=embedding_dim, num_heads=num_heads,
                                                   query_dim=embedding_dim, key_dim=embedding_dim,
                                                   value_dim=embedding_dim)
        self.masked_mah_dropout = nn.Dropout(p=dropout_p)
        self.masked_mah_norm = AddNorm(embedding_dim=embedding_dim)

        # 初始化: 多头注意层和前馈神经网络层.
        self.encoder_layer = TransformerEncoderLayer(num_heads, embedding_dim, feedforward_dim, dropout_p)

    def forward(self, encoder_out, tgt, tgt_mask=None):
        # 掩码注意力层输出.
        mml_out = self.masked_mah_layer(query_input=tgt, key_input=tgt, value_input=tgt, tgt_mask=tgt_mask)
        mml_out = self.masked_mah_dropout(mml_out)
        mml_out = self.masked_mah_norm(tgt, mml_out)  # 残差连接和标准化.

        # 注意力层和神经网络层输出.
        decoder_out = self.encoder_layer(query_input=mml_out, key_input=encoder_out, value_input=encoder_out)

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

    def forward(self, X, src_mask=None, src_key_padding_mask=None):
        """

        :param src_key_padding_mask: 源序列键填充掩码.
        :param src_mask: 原序列注意力掩码.
        :param X: shape: (batch_size,seq_len,embedding_size)
        :return: 编码器输出.
        """
        for i in range(self.num_encoder_layers):
            encoder_layer = self.encoder_layers[i]  # 第i个编码器层.
            X = encoder_layer(query_input=X, key_input=X, value_input=X,
                              src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return X


class UserTransformer(nn.Module):
    """
    Transformer模型.
    """

    def __init__(self, src_num_embeddings, tgt_num_embeddings, embedding_dim, feedforward_dim, dropout_p,
                 num_heads, num_encoder_layers, num_decoder_layers,
                 src_key_padding_mask=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        # embedding层. 权重w的shape(num_embeddings,embedding_dim)

        self.enc_embedding_layer = Embedding(num_embeddings=src_num_embeddings, embedding_dim=embedding_dim)
        # 位置编码层.
        self.enc_pe_layer = PositionalEncoding(embedding_dim=embedding_dim, dropout_p=dropout_p)
        # Transformer模型编码器.
        self.encoder = TransformerEncoder(num_encoder_layers=num_encoder_layers, num_heads=num_heads,
                                          embedding_dim=embedding_dim, feedforward_dim=feedforward_dim,
                                          dropout_p=dropout_p)

        self.dec_embedding_layer = Embedding(num_embeddings=tgt_num_embeddings, embedding_dim=embedding_dim)
        self.dec_pe_layer = self.enc_pe_layer  # 编码器和解码器位置编码相同(因为位置编码仅与位置有关).
        self.decoder = TransformerDecoder(num_decoder_layers=num_decoder_layers, num_heads=num_heads,
                                          embedding_dim=embedding_dim, feedforward_dim=feedforward_dim,
                                          dropout_p=dropout_p)
        # shape: (num_embeddings,embedding_dim).
        self.decoder_linear_weight = self.enc_embedding_layer.embedding.weight  # 权重共享.

    def forward(self, src_input, target_input, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """

        :param tgt_key_padding_mask:  目标键的填充掩码, 用于QK^T/sqrt(d_k)之后,softmax之前.
        :param tgt_mask:  目标序列的注意力掩码.
        :param src_mask: 源序列的注意力掩码.
        :param src_key_padding_mask: 源序列键填充掩码, 用于QK^T/sqrt(d_k)之后,softmax之前.
        :param src_input:  shape:(batch_size,seq_len).
        :param target_input: shape:(batch_size,seq_len,embedding_size).
        :return: 概率: shape: (batch_size,seq_len,voc_size), voc_size: 字典大小.
        """
        # 对应transformer模型架构图左侧部分.
        # embedding结果输出shape(batch_size,seq_len,embedding_size)
        encoder_embedding_out = self.enc_embedding_layer(src_input)  # embedding层输出.
        encoder_pe_out = self.enc_pe_layer(encoder_embedding_out * math.sqrt(self.embedding_dim))  # 位置编码层输出.
        encoder_out = self.encoder(encoder_pe_out, src_mask, src_key_padding_mask)  # transformer.

        # 对应transformer模型架构图右侧部分.
        decoder_embedding_out = self.dec_embedding_layer(target_input)  # decoder embedding层.
        decoder_pe_out = self.dec_pe_layer(decoder_embedding_out * math.sqrt(self.embedding_dim))  # 位置编码层.
        decoder_out = self.decoder(encoder_out, decoder_pe_out)  # shape(batch_size,seq_len,embedding_dim).
        # 结果shape (batch_size,seq_len,num_embeddings).
        out = decoder_out @ self.decoder_linear_weight.t()  # 未经过softmax归一化的logits值.

        # out = torch.softmax(out, dim=-1)  # decoder的softmax层.

        return out
