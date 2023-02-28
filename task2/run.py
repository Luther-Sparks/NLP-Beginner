import csv
import random
from feature import RandomEmbedding, GloveEmbedding
from comparison_plot_batch import NN_embedding_plot
import torch 

with open('./sentiment-analysis-on-movie-reviews/train.tsv') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    tsv_reader = list(tsv_reader)

with open('./glove.6B/glove.6B.50d.txt', 'rb') as f:
    trained_words = f.readlines()

# 用GloVe创建词典
trained_dict = dict()
for word in trained_words:
    word_vector = word.split()  # 将单词与参数分割,并以list形式存储
    trained_dict[word_vector[0].decode('utf-8').upper()] = [float(i) for i in word_vector[1:]]

# 超参初始化
iter_times = 50
learning_rate = 0.001

data = tsv_reader[1:]
batch_size = 500

# 随机初始化
random.seed(2021)
random_embedding = RandomEmbedding(data=data)
random_embedding.getWords()             # 找到所有单词并标记ID
random_embedding.getId()                # 找到每个句子拥有的单词ID

# 预训练GloVe模型初始化
random.seed(2021)
glove_embedding = GloveEmbedding(data=data, trained_dict=trained_dict)
glove_embedding.getWords()
glove_embedding.getId()

NN_embedding_plot(random_embedding, glove_embedding, learning_rate, batch_size, iter_times)