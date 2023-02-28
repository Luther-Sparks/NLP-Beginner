import numpy
import random


class softmax:
    def __init__(self, sample, typenum, feature):
        """Data Initialization

        Args:
            sample (int): [the number of sample of train data set]
            typenum (int): [the number of type]
            feature (int): [the number of feature, also the length of the one-hot vector]
            W (feature * typenum matrix): [matrix of W]
        """
        self.sample = sample                                        
        self.typenum = typenum
        self.feature = feature
        self.W = numpy.random.randn(self.feature, self.typenum)

    def softmax_calculation(self, x):
        """calculate softmax(x)

        Args:
            x (vector): [description]
        """
        exp = numpy.exp(x - numpy.max(x))
        return exp / exp.sum()

    def softmax_all(self, wtx):
        """calculate softmax(wtx)

        Args:
            wtx (matrix): [description]
        """
        delta_wtx = wtx - numpy.max(wtx, axis=1, keepdims=True)
        exp = numpy.exp(delta_wtx)
        return exp / numpy.sum(wtx, axis=1, keepdims=True)

    def int2one_hot(self, x):
        """transform an 'int' into a one-hot vector

        Args:
            x (int): [description]
        """
        one_hot = numpy.zeros(self.typenum, dtype=int)
        one_hot[x] = 1
        return one_hot.reshape(-1,1)

    def prediction(self, X):
        """Given matrix X, predict the category

        Args:
            X (matrix): [description]
        """
        prob = self.softmax_all(X.dot(self.W))
        return prob.argmax(axis=1)

    def accuracy(self, train, train_y, test, test_y):
        pred_train = self.prediction(train)
        train_accuracy = sum([train_y == pred_train[i] for i in range(len(train))]) / len(train)
        pred_test = self.prediction(test)
        test_accuracy = sum([test_y == pred_test[i] for i in range(len(test))]) / len(test)
        print('train_accuracy:', train_accuracy)
        print('test_accuracy:', test_accuracy)
        return train_accuracy, test_accuracy
    
    def regression(self, X, y, alpha, times, strategy="mini", batch_size=100):
        """softmax regression

        Args:
            X (matrix): input
            y (vector): output
            alpha (float): learning rate
            times (int): train times
            strategy (str, optional): mode. Defaults to "mini".
            batch_size (int, optional): batch size. Defaults to 100.
        """
        if self.sample != len(X) or self.sample != len(y):
            raise Exception("Sample size doesn't match")
        
        if strategy == 'mini':
            for i in range(times):
                gradient = numpy.zeros((self.feature, self.typenum))
                for j in range(batch_size):
                    k = random.randint(0, self.sample - 1)
                    yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                    gradient += X[k].reshape(-1, 1).dot((self.int2one_hot(y[k]) - yhat).T)
                self.W += alpha / batch_size * gradient
        elif strategy == 'shuffle':
            for i in range(times):
                k = random.randint(0, self.sample - 1)  # choose a sample randomly
                yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                gradient = X[k].reshape(-1, 1).dot((self.int2one_hot(y[k]) - yhat).T)
                self.W += alpha * gradient
        elif strategy == "batch":
            for i in range(times):
                gradient = numpy.zeros((self.feature, self.typenum))
                for j in range(self.sample):
                    yhat = self.softmax_calculation(self.W.T.dot(X[j].reshape(-1,1)))
                    gradient += X[j].reshape(-1, 1).dot((self.int2one_hot(y[j]) - yhat).T)
                self.W += alpha / self.sample * gradient
        else:
            raise Exception('Unknown strategy')