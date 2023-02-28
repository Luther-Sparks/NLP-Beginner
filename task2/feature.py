import random
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.nn.utils.rnn import pack_sequence, pad_sequence

def splitData(data_set, split_rate=0.2):
    train = list()
    test = list()
    for data in data_set:
        if random.random() > split_rate:
            train.append(data)
        else:
            test.append(data)
    return train, test


class RandomEmbedding():
    def __init__(self, data, split_rate=0.2):
        self.words_dict = dict()
        data.sort(key=lambda x:len(x[2].split()))       # 由于句子的特征与句子的长度无关，所以依据句子的长度来进行排序可以达到随机的目的
        self.data = data
        self.words_length = 0
        self.train, self.test = splitData(data, split_rate)
        self.train_y = [int(item[3]) for item in self.train]    
        self.test_y  = [int(item[3]) for item in self.test]
        self.train_matrix = list()
        self.test_matrix  = list()
        self.longest = 0                                # 记录最长的单词

    def getWords(self):
        for item in self.data:
            sentence = item[2]
            sentence = sentence.upper()
            words = sentence.split()
            for word in words:
                if word not in self.words_dict:
                    self.words_dict[word] = len(self.words_dict) + 1    # 加上第0个的padding
        self.words_length = len(self.words_dict)

    def getId(self):
        for item in self.train:
            sentence = item[2]
            sentence = sentence.upper()
            words = sentence.split()
            id_list = [self.words_dict[word] for word in words]
            self.longest = max(self.longest, len(id_list))
            self.train_matrix.append(id_list)
        
        for item in self.test:
            sentence = item[2]
            sentence = sentence.upper()
            words = sentence.split()
            id_list = [self.words_dict[word] for word in words]
            self.longest = max(self.longest, len(id_list))
            self.test_matrix.append(id_list)
        self.words_length += 1                                      # 加上第0个的padding


class GloveEmbedding():
    def __init__(self, data, trained_dict, split_rate=0.2):
        self.trained_dict = trained_dict
        self.data = data
        self.words_dict = dict()
        self.words_length = 0
        self.train, self.test = splitData(self.data, split_rate)
        self.train_y = [int(item[3]) for item in self.train]
        self.test_y = [int(item[3]) for item in self.test]
        self.train_matrix = list()
        self.test_matrix  = list()
        self.longest = 0                        # 单词数最多的句子
        self.trained_words = list()             # 抽取出预训练模型中用到的单词

    def getWords(self):
        self.trained_words.append([0]*50)
        for item in self.data:
            sentence = item[2]
            sentence = sentence.upper()
            words = sentence.split()
            for word in words:
                if word not in self.words_dict:
                    self.words_dict[word] = len(self.words_dict) + 1
                    if word not in self.trained_dict:
                        self.trained_words.append([0]*50)       # 预训练模型中未出现过该单词，故初始化为0向量
                    else:
                        self.trained_words.append(self.trained_dict[word])
        self.words_length = len(self.words_dict)            # 未包含padding的ID:0

    def getId(self):
        for item in self.train:
            sentence = item[2]
            sentence = sentence.upper()
            words = sentence.split()
            id_list = [self.words_dict[word] for word in words]
            self.longest = max(self.longest, len(id_list))
            self.train_matrix.append(id_list)
        
        for item in self.test:
            sentence = item[2]
            sentence = sentence.upper()
            words = sentence.split()
            id_list = [self.words_dict[word] for word in words]
            self.longest = max(self.longest, len(id_list))
            self.test_matrix.append(id_list)
        self.words_length += 1


class myDataSet(Dataset):
    def __init__(self, sentence, emotion):
        self.sentence = sentence
        self.emotion = emotion
    
    def __getitem__(self, index):
        return self.sentence[index], self.emotion[index]

    def __len__(self):
        return len(self.emotion)

def collate_fn(batch_data):
    sentence, emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)

def getBatch(sentence, emotion, batch_size):
    dataset = myDataSet(sentence, emotion)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader
