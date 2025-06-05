import os
from io import open
import torch


class Dictionary(object):
    """
    字典.
    """

    def __init__(self):
        """
        初始化方法.
        """
        self.word2idx = {}  # 根据word找id.
        self.idx2word = []  # 根据id找word, 它其实构成了字典.

    def add_word(self, word):
        """
        将word加入词典中.
        :param word: 待加入词典中的word.
        :return: 返回word在词典的id.
        """
        if word not in self.word2idx:  # 如果词不在字典中, 则加入.
            self.idx2word.append(word)  # 将word放入列表中, 列表的索引就是word的id.
            self.word2idx[word] = len(self.idx2word) - 1  # 建立word到id的映射.

        return self.word2idx[word]

    def __len__(self):
        """
        字典大小.
        :return: 字典大小.
        """
        return len(self.idx2word)


class Corpus(object):
    """
    语料.
    """

    def __init__(self, path):
        """
        初始化方法.
        :param path: 语料路径（数据集的目录）.
        """
        self.dictionary = Dictionary()  # 初始化字典.
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # 训练集分词.
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))  # 验证集分词.
        self.test = self.tokenize(os.path.join(path, 'test.txt'))  # 测试集分词.

    def tokenize(self, path):
        """
        分词.
        :param path: 语料路径.
        :return:
        """
        assert os.path.exists(path)  # 断言语料路径是否存在.
        # 将word加入字典中.
        with open(path, 'r', encoding="utf8") as f:  # 打开文件.
            idss = []  # 序列的列表. 每个元素表示一个序列.
            for line in f:  # 逐行遍历.
                # 将一行按照空白字符（包括\n \r \t \f 和 空格）切分. 这里将一行看做一个序列, <eos>标记一个序列结束.
                words = line.split() + ['<eos>']
                ids = []  # 一个序列的id列表.
                for word in words:
                    self.dictionary.add_word(word)  # 将word逐个加入字典中.
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))  # 一个序列的id转换成一个张量.
            ids = torch.cat(idss)  # 将一个张量列表连接为一个张量.

        return ids

    def get_dic_token_num(self):
        """
        返回字典的token数量.
        :return:
        """
        return len(self.dictionary)


def batchify(raw_data, bsz, device='cpu'):
    """
    数据分批.
    如果对26个字母构成的序列进行分批，批次大小为4, 如果丢弃最后两个字母，则可分成6个批次. 整个过程如下：
    (1) 原始序列
            raw_data = [a,b,...,x,y,z].
    (2) narrow操作
            data = [a,b,...,x].
    (3) view操作
            data = [[a,b,c,d,e,f]
                    [g,h,i,j,k,l]
                    [m,n,o,p,q,r]
                    [s,t,u,v,w,x]]
        形状为(批次大小，每批序列长度)=(4,6).
    (4) t()转置
            ┌ a g m s ┐
            │ b h n t │
            │ c i o u │
            │ d j p v │
            │ e k q w │
            └ f l r x ┘.
        形状为(每批序列长度，批次大小)=(6,4).
    :param device: cpu或cuda.
    :param raw_data: 数据序列.
    :param bsz: 批大小.
    :return: 分批后的数据，形状为(每批序列长度，批次大小).
    """
    # 数据大小整除批大小, //表示整除, 例如: 7//3=2.
    seq_len = raw_data.size(0) // bsz
    # 沿着数据张量的第0维从start开始截取，截取的word个数为nbatch * bsz.
    data = raw_data.narrow(dim=0, start=0, length=seq_len * bsz)
    data = data.view(bsz, -1)  # 形状(批大小，批次数）.
    data = data.t().contiguous()  # 转置. 形状(批次数，批大小).

    return data.to(device)


def get_batch(source, i, seq_len):
    """
    获取批次数据.

    :param source: 源数据.
    :param i: 批次号.
    :param seq_len: 序列长度.
    :return:
    """
    # 如果source长度100, seq_len=35, 则source分为三个序列，长度分别为35,35,30.
    seq_len = min(seq_len, len(source) - 1 - i)
    src_data = source[i:i + seq_len]  # 源序列
    target_data = source[i + 1:i + 1 + seq_len]  # 目标序列, 与源序列相同形状.

    return src_data, target_data
