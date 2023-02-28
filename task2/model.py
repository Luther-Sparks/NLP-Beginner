import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class myRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, words_len, typenum=5, weight=None, layer=1, nonlinearity='tanh', batch_first=True, drop_out=0.5):
        super(myRNN, self).__init__()
        self.feature_len = feature_len      # D的大小
        self.hidden_len = hidden_len        # l_h的大小
        self.words_len = words_len          # 单词个数(包括padding)
        self.layer = layer                  # 隐藏层层数
        self.dropout = nn.Dropout(drop_out)
        if weight is None:                  # 随机初始化
            x = nn.init.xavier_normal_(torch.Tensor(self.words_len, feature_len))
            # xavier初始化方法中服从正态分布,通过网络层,输入和输出的方差相同
            # self.embedding = nn.Embedding(num_embeddings=words_len, embedding_dim=feature_len, _weight=x)
            self.embedding = nn.Embedding(num_embeddings=words_len, embedding_dim=feature_len, _weight=x).cuda()
        else:
            # self.embedding = nn.Embedding(num_embeddings=words_len, embedding_dim=feature_len, _weight=weight)
            self.embedding = nn.Embedding(num_embeddings=words_len, embedding_dim=feature_len, _weight=weight).cuda()
        # 用nn.Module的内置函数定义隐藏层
        # self.rnn = nn.RNN(input_size=feature_len, hidden_size=hidden_len, num_layers=layer, nonlinearity=nonlinearity,
        #     batch_first=batch_first, dropout=drop_out)
        self.rnn = nn.RNN(input_size=feature_len, hidden_size=hidden_len, num_layers=layer, nonlinearity=nonlinearity,
            batch_first=batch_first, dropout=drop_out) .cuda()

        # 全连接层
        # self.fc = nn.Linear(in_features=hidden_len, out_features=typenum)  
        self.fc = nn.Linear(in_features=hidden_len, out_features=typenum).cuda()  

    def forward(self, data):
        # data: 数据(维度为[batch_size, len(sentence)])
        # data = torch.Tensor(data)
        data = torch.LongTensor(data).cuda()
        batch_size = data.size(0)
        # 经过词嵌入后,维度为[batch_size, len(sentence), d]
        output = self.embedding(data)   # 词嵌入
        output = self.dropout(output)
        # 正态分布初始化h_0
        # h0 = torch.randn(self.layer, batch_size, self.hidden_len)
        # h0 = torch.autograd.Variable(h0)
        h0 = torch.randn(self.layer, batch_size, self.hidden_len).cuda()
        h0 = torch.autograd.Variable(h0).cuda()
        # 经过dropout后不变,经过隐藏层后,维度为[1, batch_size, l_h]
        _, hn = self.rnn(output, h0)    # 隐藏层计算
        # 经过全连接层后,维度变为[1, batch_size, self.typenum=5]
        output = self.fc(hn).squeeze(0)
        # 去掉第0维度,返回[batch_size, typenum]的数据
        return output


class myCNN(nn.Module):
    def __init__(self, feature_len, words_len, longest, typenum=5, weight=None, dropout=0.5):
        super(myCNN, self).__init__()
        self.feature_len = feature_len          # d的大小
        self.words_len   = words_len            # 单词数目
        self.longest     = longest              # 最长句子的单词数目
        self.dropout     = nn.Dropout(dropout)  # dropout层
        if weight is None:  # 随机初始化
            data = nn.init.xavier_normal_(torch.Tensor(self.words_len, self.feature_len))
            # self.embedding = nn.Embedding(num_embeddings=self.words_len, embedding_dim=self.feature_len, _weight=data) 
            self.embedding = nn.Embedding(num_embeddings=self.words_len, embedding_dim=self.feature_len, _weight=data).cuda()
        else:               # GloVe初始化
            # self.embedding = nn.Embedding(num_embeddings=self.words_len, embedding_dim=self.feature_len, _weight=weight) 
            self.embedding = nn.Embedding(num_embeddings=self.words_len, embedding_dim=self.feature_len, _weight=weight).cuda()
        conv_size = 4
        self.conv = list()
        for i in range(conv_size):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.longest, kernel_size=(i+2, self.feature_len), padding=(math.ceil((i+2-1)/2), 0))
            , nn.ReLU()).cuda())
        # self.fc = nn.Linear(conv_size*self.longest, typenum)
        self.fc = nn.Linear(conv_size*self.longest, typenum).cuda()

    def forward(self, x):
        # x: 数据,维度[batch_size, sentence_length]
        # x = torch.Tensor(x)
        x = torch.LongTensor(x).cuda()
        # 经过词嵌入后,维度为[batch_size, 1(C_in channel个数), sentence_length(特征图的高), d(特征图的宽)]
        output = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.feature_len)
        output = self.dropout(output)

        # 经过卷积后,维度为[batch_size, l_l(C_out channel个数), sentence_length+(i+1)%2(包括了padding造成的行数扩张)(特征图的高), 1(特征图的宽)]
        # 挤掉第三维度
        conv = [_conv(output).squeeze(3) for _conv in self.conv]

        # 分别对几个conv结果的第二维进行pooling,得到4个[batch_size, l_l, 1]向量
        # 挤掉第二维度为[batch_size, l_l*conv_size]
        pool = [F.max_pool1d(_conv, _conv.shape[2]) for _conv in conv]

        # 拼接得到[batch_size, l_l*conv_size, 1]的向量
        # 挤掉第二维度为[batch_size, l_l*conv_size]
        pool = torch.cat(pool, 1).squeeze(2)

        #经过全连接层后,维度为[batch_size, typenum=5]
        output = self.fc(pool)
        return output

        