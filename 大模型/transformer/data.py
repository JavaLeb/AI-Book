import itertools
import os
from io import open
import torch
import jieba
import nltk
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from config import PytorchTransConfig as Config


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
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.key_tokens = [pad_token, bos_token, eos_token, unknown_token]
        self.add_tokens(self.key_tokens)
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = self._get_key_ids(self.key_tokens)
        if reserved_tokens:
            self.add_tokens(reserved_tokens)  # 将保留词元加入词典.

    def add_token(self, token):
        """
        将word加入词典中.
        :param token: 待加入词典中的word.
        :return: 返回word在词典的id.
        """
        if token not in self.token2idx:  # 如果词不在字典中, 则加入.
            self.idx2token.append(token)  # 将word放入列表中, 列表的索引就是word的id.
            self.token2idx[token] = len(self.idx2token) - 1  # 建立word到id的映射.

        return self.token2idx[token]

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def _get_key_ids(self, key_tokens):
        return [self.token2idx[token] for token in key_tokens]

    def __len__(self):
        """
        字典大小.
        :return: 字典大小.
        """
        return len(self.idx2token)


class Corpus(object):
    """
    语料.
    """

    def __init__(self, path, num_line: int = -1, device='cpu'):
        """
        初始化方法.
        :param path: 语料路径（数据集的目录）.
        """

        self.dictionary = Dictionary()  # 初始化字典.
        self.num_line = num_line
        self.src_max_seq_len = -1
        self.tgt_max_seq_len = -1
        self.device = device
        self.data_path = path
        self.data_set = self.tokenize()

    def tokenize(self):
        """
        分词.
        :return:
        """
        assert os.path.exists(self.data_path)  # 断言语料路径是否存在.
        with open(self.data_path, 'r', encoding="utf8") as f:  # 打开文件.
            if self.num_line > 0:
                lines = []
                i = 0
                while i < self.num_line:
                    lines.append(f.readline().rstrip('\n'))
                    i += 1
            else:
                text = f.read()
                lines = text.split('\n')

        data_set = []
        bos_token = self.dictionary.bos_token
        eos_token = self.dictionary.eos_token
        for line in lines:
            if len(line) == 0:  # 排除空白行.
                continue
            part = line.split('\t')  # 按照tab切分.
            src_sent, tgt_sent = part[0], part[1]  # 源句子和目标句子.
            src_tokens = nltk.word_tokenize(src_sent) + [eos_token]  # 英文分词.
            tgt_tokens = [bos_token] + jieba.lcut(tgt_sent) + [eos_token]  # 中文分词.
            if len(src_tokens) > self.src_max_seq_len:
                self.src_max_seq_len = len(src_tokens)
            if len(tgt_tokens) > self.tgt_max_seq_len:
                self.tgt_max_seq_len = len(tgt_tokens)
            src_ids = self._tokens2ids(src_tokens)  # token数字化.
            tgt_ids = self._tokens2ids(tgt_tokens)  # token数字化.
            data_set.append((src_ids, tgt_ids))

        result_data = []
        for (src_ids, tgt_ids) in data_set:
            src_ids = nn.functional.pad(src_ids, (0, self.src_max_seq_len - len(src_ids)),
                                        value=self.dictionary.pad_id)
            tgt_ids = nn.functional.pad(tgt_ids, (0, self.tgt_max_seq_len - len(tgt_ids)),
                                        value=self.dictionary.pad_id)
            result_data.append((src_ids, tgt_ids))

        return result_data

    def _tokens2ids(self, tokens):
        ids = []  # 一个序列的id列表.
        for token in tokens:
            self.dictionary.add_token(token)
            ids.append(self.dictionary.token2idx[token])
        return torch.tensor(ids, dtype=torch.int64, device=self.device)

    def ids2tokens(self, token_ids):
        tokens = []
        for token_id in token_ids:
            token = self.dictionary.idx2token[token_id]
            tokens.append(token)
        return tokens

    def get_dic_size(self):
        """
        返回字典的token数量.
        :return:
        """
        return len(self.dictionary)

    def get_max_seq_len(self):
        return max(self.src_max_seq_len, self.tgt_max_seq_len)


class DataSetProcessor:
    def __init__(self, data_set):
        self.val_dataset = None
        self.train_dataset = None
        self.data_set = data_set

    def split(self, train_size=0.8):
        train_size = int(train_size * len(self.data_set))
        val_size = len(self.data_set) - train_size
        self.train_dataset, self.val_dataset = random_split(self.data_set, [train_size, val_size])

        return self.train_dataset, self.val_dataset

    def batchify(self, batch_size=10, shuffle=True):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader


if __name__ == '__main__':
    torch.manual_seed(42)
    corpus = Corpus(Config.data_path, device=Config.device)
    vocab_size = corpus.get_dic_size()
    max_seq_len = corpus.get_max_seq_len()
    dataset_processor = DataSetProcessor(corpus.data_set)
    train_dataset, val_dataset = dataset_processor.split(train_size=0.8)
    train_loader, val_loader = dataset_processor.batchify(batch_size=Config.batch_size)
    print()
