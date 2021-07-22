from typing import Sequence
import numpy
import random


def splitData(data_set, split_rate=0.2, max_item=10000):
    """[summary] Split the dataset into two parts: train and test using the split_rate

    Args:
        data_set ([list])
        split_rate (float, optional): Defaults to 0.2.
        max_item (int, optional): Defaults to 10000.

    Returns:
        train [list]: [train data]
        test [list]: [test data]
    """
    train = list()
    test = list()
    i = 0
    for data in data_set:
        i += 1
        if random.random() > split_rate:
            train.append(data)
        else:
            test.append(data)
        if i > max_item:
            break
        
    return train, test

class BoW:
    def __init__(self, data, max_item=10000):
        self.data = data
        self.max_item = max_item
        self.words_dict = dict()        # feature table
        self.len = 0                    # record the number of feature
        self.train, self.test = splitData(self.data, split_rate=0.2, max_item=self.max_item)
        self.train_y = [int(item[3]) for item in self.train]
        self.test_y  = [int(item[3]) for item in self.test]
        self.train_matrix = None
        self.test_matrix = None

    def get_words(self):
        for item in self.data:
            sentence = str(item[2])
            sentence = sentence.upper()
            words = sentence.split()
            for word in words:
                if word not in self.words_dict:
                    self.words_dict[word] = len(self.words_dict)
        self.len = len(self.words_dict)
        self.train_matrix = numpy.zeros((len(self.train), self.len))
        self.test_matrix = numpy.zeros((len(self.test), self.len))
    
    def get_matrix(self):
        for i in range(len(self.train)):
            sentence = str(self.train[i][2])
            words = sentence.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.words_dict[word]] = 1
        for i in range(len(self.test)):
            sentence = str(self.test[i][2])
            words = sentence.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.words_dict[word]] = 1


class N_Gram:
    def __init__(self, data, dimension, max_item):
        self.data = data
        self.max_item = max_item
        self.words_dict = dict()        # feature table
        self.len = 0                    # record the number of feature
        self.dimension = dimension      # dimension as known the N
        self.train, self.test = splitData(self.data, split_rate=0.2, max_item=self.max_item)
        self.train_y = [int(item[3]) for item in self.train]
        self.test_y  = [int(item[3]) for item in self.test]
        self.train_matrix = None        # feature vectors of train data set
        self.test_matrix = None         # feature vectors of test data set

    def get_words(self):
        for d in range(1, self.dimension + 1):
            for item in self.data:
                sentence = str(item[2])
                sentence = sentence.upper()
                words = sentence.split()
                for i in range(len(words) - d + 1):
                    n_gram_feature = words[i: i+d]
                    n_gram_feature = "_".join(n_gram_feature)
                    if n_gram_feature not in self.words_dict:
                        self.words_dict[n_gram_feature] = len(self.words_dict)
        self.len = len(self.words_dict)
        self.train_matrix = numpy.zeros((len(self.train), self.len))
        self.test_matrix = numpy.zeros((len(self.test), self.len))

    def get_matrix(self):
        for d in range(1, self.dimension + 1):
            for i in range(len(self.train)):
                sentence = str(self.train[i][2])
                sentence = sentence.upper()
                words = sentence.split()
                for j in range(len(words) - d + 1):
                    n_gram_feature = words[j: j + d]
                    n_gram_feature = "_".join(n_gram_feature)
                    self.train_matrix[i][self.words_dict[n_gram_feature]] = 1
        for d in range(1, self.dimension + 1):
            for i in range(len(self.test)):
                sentence = str(self.test[i][2])
                sentence = sentence.upper()
                words = sentence.split()
                for j in range(len(words) - d + 1):
                    n_gram_feature = words[j: j + d]
                    n_gram_feature = "_".join(n_gram_feature)
                    self.test_matrix[i][self.words_dict[n_gram_feature]] = 1