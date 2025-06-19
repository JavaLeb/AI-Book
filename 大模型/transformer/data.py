import os
from io import open
import torch
import jieba
import nltk
from torch.utils.data import DataLoader, random_split
from torch import nn
import random


class Dictionary(object):
    """
    字典.
    """

    def __init__(self, pad_token='<pad>', bos_token='<bos>',
                 eos_token='<eos>', unknown_token='<unk>', reserved_tokens: list = None):
        """
        初始化方法.
        """
        self.token2idx = {}  # 根据word找id.
        self.idx2token = []  # 根据id找word, 它其实构成了字典.
        self.bos_token, self.eos_token = bos_token, eos_token
        self.key_tokens = [pad_token, bos_token, eos_token, unknown_token]
        self.key_token_ids = self.add_tokens(self.key_tokens)
        self.pad_id = self.key_token_ids[0]
        if reserved_tokens:
            self.add_tokens(reserved_tokens)  # 将保留词元加入词典.

    def add_token(self, token: str):
        """
        将word加入词典中.
        :param token: 待加入词典中的word.
        :return: 返回word在词典的id.
        """
        if token not in self.token2idx:  # 如果词不在字典中, 则加入.
            self.idx2token.append(token)  # 将word放入列表中, 列表的索引就是word的id.
            self.token2idx[token] = len(self.idx2token) - 1  # 建立word到id的映射.

        return self.token2idx[token]

    def add_tokens(self, tokens: list):
        """
        批量添加token.
        :param tokens: token列表.
        :return: tokens对应的id列表.
        """
        return [self.add_token(token) for token in tokens]

    def ids2tokens(self, token_ids):
        """
        ids转换为token列表.
        :param token_ids: token_id列表.
        :return: token列表.
        """

        return [self.idx2token[token_id] for token_id in token_ids]

    def tokens2ids(self, tokens):
        """
        token转换为id列表.
        :param tokens: token列表.
        :return: token_id列表.
        """
        return [self.token2idx[token] for token in tokens]

    def size(self):
        """
        字典大小.
        :return: 字典大小.
        """
        return len(self.idx2token)


class Corpus(object):
    """
    语料.
    """

    def __init__(self, path, num_line: int = -1, device='cpu', max_seq_len=50):
        """
        初始化方法.
        :param path: 语料路径（数据集的目录）.
        """

        self.dictionary = Dictionary()  # 初始化字典.
        self.num_line = num_line  # 读取文件的行数.
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.device = device  # 设备.
        self.data_path = path  # 数据路径.
        self.data_set = self.process()  # 分词.

    def read(self):
        assert os.path.exists(self.data_path)  # 断言语料路径是否存在.
        with open(self.data_path, 'r', encoding="utf8") as f:  # 打开文件.
            if self.num_line > 0:  # readline()会读取换行符, rstrip()删除右边的换行符.
                lines = [f.readline().rstrip('\n') for _ in range(self.num_line)]
            else:
                lines = f.read().split('\n')

        return lines

    def tokenize(self, src_sent, tgt_sent):
        """
        分词
        :param src_sent: 源语言句子.
        :param tgt_sent: 目标语言句子.
        :return:
        """

        src_tokens = nltk.word_tokenize(src_sent)  # 英文分词.
        tgt_tokens = jieba.lcut(tgt_sent)  # 中文分词.

        return src_tokens, tgt_tokens

    def process(self):
        """
        token处理.
        :return:
        """
        lines = self.read()

        data_set = []
        bos_token = self.dictionary.bos_token
        eos_token = self.dictionary.eos_token
        for line in lines:
            if len(line) == 0:  # 排除空白行.
                continue
            part = line.split('\t')  # 按照tab切分.
            src_sent, tgt_sent = part[0], part[1]  # 源句子和目标句子.
            src_tokens, tgt_tokens = self.tokenize(src_sent, tgt_sent)
            src_tokens = src_tokens + [eos_token]
            tgt_tokens = [bos_token] + tgt_tokens + [eos_token]
            src_ids = self.add_padding(src_tokens)  # 添加tokens, 并返回ids.
            tgt_ids = self.add_padding(tgt_tokens)
            data_set.append((src_ids, tgt_ids, src_sent, tgt_sent))

        return data_set

    def add_padding(self, tokens):
        """
        添加token到词典, 转为索引, 并填充.
        :param tokens:
        :return:
        """

        token_ids = self.dictionary.add_tokens(tokens)  # 添加到词典, 并返回ids.
        token_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.device)  # 转化为张量.
        # padding.
        token_ids = nn.functional.pad(
            token_ids, (0, self.max_seq_len - len(token_ids)), value=self.dictionary.pad_id)

        return token_ids


class DataSetProcessor:
    """
    数据集处理器.
    """

    def __init__(self, data_set):
        """
        数据集.
        :param data_set:
        """
        self.data_set = data_set  # 数据集.
        self.train_dataset = None  # 训练集.
        self.val_dataset = None  # 验证集.
        self.test_dataset = None  # 测试集.

    def split(self, train_size=0.8, test_size=5):
        """
        数据集切分.
        :param train_size: 训练集比例.
        :param test_size: 测试集数量.
        :return:
        """

        train_size = int(train_size * len(self.data_set))  # 训练集样本数.
        val_size = len(self.data_set) - train_size  # 剩余作为验证集.
        self.train_dataset, self.val_dataset = random_split(self.data_set, [train_size, val_size])
        # 随机选择几个测试集样本.
        indices = random.sample(range(len(self.data_set)), test_size)
        self.test_dataset = [self.data_set[i] for i in indices]

        return self.train_dataset, self.val_dataset, self.test_dataset

    def batchify(self, batch_size=10, shuffle=True):
        """
        数据集分批.
        :param batch_size: 批次大小.
        :param shuffle: 是否打乱数据.
        :return:
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle)
        # 测试集只生产一批数据.
        test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=shuffle)

        return train_loader, val_loader, test_loader
