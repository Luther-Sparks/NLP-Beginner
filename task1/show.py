import matplotlib.pyplot as plt
from softmax import softmax
from feature import BoW, N_Gram

def alpha_gradient_plot(bow:BoW, n_gram:N_Gram, total_times, batch_size):
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    # BoW
    
    # shuffle
    shuffle_train = list()
    shuffle_test = list()
    for alpha in alphas:
        soft = softmax(len(bow.train), 5, bow.len)
        soft.regression(bow.train_matrix, bow.train_y, alpha, total_times, "shuffle")
        train_accuracy, test_accuracy = soft.accuracy(bow.train_matrix, bow.train_y, bow.test_matrix, bow.test_y)
        shuffle_train.append(train_accuracy)
        shuffle_test.append(test_accuracy)

    # batch
    batch_train = list()
    batch_test = list()
    for alpha in alphas:
        soft = softmax(len(bow.train), 5, bow.len)
        soft.regression(bow.train_matrix, bow.train_y, alpha, int(total_times/ bow.max_item), "batch")
        train_accuracy, test_accuracy = soft.accuracy(bow.train_matrix, bow.train_y, bow.test_matrix, bow.test_y)
        batch_train.append(train_accuracy)
        batch_test.append(test_accuracy)

    # mini-batch
    mini_batch_train = list()
    mini_batch_test = list()
    for alpha in alphas:
        soft = softmax(len(bow.train), 5, bow.len)
        soft.regression(bow.train_matrix, bow.train_y, alpha, int(total_times/ batch_size), "mini")
        train_accuracy, test_accuracy = soft.accuracy(bow.train_matrix, bow.train_y, bow.test_matrix, bow.test_y)
        mini_batch_train.append(train_accuracy)
        mini_batch_test.append(test_accuracy)


    plt.subplot(2, 2, 1)
    plt.semilogx(alphas, shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_train, 'g--', label='batch')
    plt.semilogx(alphas, mini_batch_train, 'b--', label='mini_batch')
    plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_batch_train, 'b^-')
    plt.legend()
    plt.title("BoW -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 2)
    plt.semilogx(alphas, shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_test, 'g--', label='batch')
    plt.semilogx(alphas, mini_batch_test, 'b--', label='mini_batch')
    plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_batch_test, 'b^-')
    plt.legend()
    plt.title("BoW -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    # N-Gram
    
    # shuffle
    shuffle_train = list()
    shuffle_test = list()
    for alpha in alphas:
        soft = softmax(len(n_gram.train), 5, n_gram.len)
        soft.regression(n_gram.train_matrix, n_gram.train_y, alpha, total_times, "shuffle")
        train_accuracy, test_accuracy = soft.accuracy(n_gram.train_matrix, n_gram.train_y, n_gram.test_matrix, n_gram.test_y)
        shuffle_train.append(train_accuracy)
        shuffle_test.append(test_accuracy)

    # batch
    batch_train = list()
    batch_test = list()
    for alpha in alphas:
        soft = softmax(len(n_gram.train), 5, n_gram.len)
        soft.regression(n_gram.train_matrix, n_gram.train_y, alpha, int(total_times/ n_gram.max_item), "batch")
        train_accuracy, test_accuracy = soft.accuracy(n_gram.train_matrix, n_gram.train_y, n_gram.test_matrix, n_gram.test_y)
        batch_train.append(train_accuracy)
        batch_test.append(test_accuracy)

    # mini-batch
    mini_batch_train = list()
    mini_batch_test = list()
    for alpha in alphas:
        soft = softmax(len(n_gram.train), 5, n_gram.len)
        soft.regression(n_gram.train_matrix, n_gram.train_y, alpha, int(total_times/ batch_size), "mini")
        train_accuracy, test_accuracy = soft.accuracy(n_gram.train_matrix, n_gram.train_y, n_gram.test_matrix, n_gram.test_y)
        mini_batch_train.append(train_accuracy)
        mini_batch_test.append(test_accuracy)

    plt.subplot(2, 2, 3)
    plt.semilogx(alphas, shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_train, 'g--', label='batch')
    plt.semilogx(alphas, mini_batch_train, 'b--', label='mini_batch')
    plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_batch_train, 'b^-')
    plt.legend()
    plt.title("N-Gram -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 4)
    plt.semilogx(alphas, shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_test, 'g--', label='batch')
    plt.semilogx(alphas, mini_batch_test, 'b--', label='mini_batch')
    plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_batch_test, 'b^-')
    plt.legend()
    plt.title("N-Gram -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()