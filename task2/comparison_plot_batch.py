import matplotlib.pyplot as plt
from torch import optim
import torch
import torch.nn.functional as F
from feature import getBatch
from model import myCNN, myRNN

def NN_embedding(model, train, test, learning_rate, iter_times):
    # 定义优化器(求参数)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 定义损失函数
    loss_func = F.cross_entropy
    # 损失值记录
    train_loss_record = list()
    test_loss_record = list()
    long_loss_record = list()
    # 准确率记录
    train_record = list()
    test_record  = list()
    long_record  = list()
    #训练阶段
    for iteration in range(iter_times):
        model.train()
        print(train)
        for batch in train:
            x, y = batch                        # 取一个batch
            y = y.cuda()
            pred = model(x).cuda()              # 计算输出
            optimizer.zero_grad()               # 梯度初始化
            loss = loss_func(pred, y).cuda()    # 损失值计算
            loss.backward()                     # 梯度反向传播
            optimizer.step()                    # 更新参数

        # 测试模式
        model.eval()
        # 本轮正确率记录
        train_acc = list()
        test_acc  = list()
        long_acc  = list()
        length    = 20
        # 本轮损失值计算
        train_loss = 0
        test_loss  = 0
        long_loss  = 0
        for batch in train:
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            loss = loss_func(pred, y).cuda()
            train_loss += loss.item()          # 损失值累计
            _, y_pre = torch.max(pred, -1)
            # 计算本次batch的准确率
            acc = torch.mean((torch.tensor(y_pre==y, dtype=torch.float)))
            train_acc.append(acc)

        for batch in test:
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            loss = loss_func(pred, y).cuda()
            test_loss += loss.item()
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre==y, dtype=torch.float)))
            test_acc.append(acc)
            if len(x[0]) > length:
                long_acc.append(acc)
                long_loss += loss.item()

        trains_acc = sum(train_acc) / len(train_acc)
        tests_acc  = sum(test_acc) / len(test_acc)
        longs_acc  = sum(long_acc) / len(long_acc)

        train_loss_record.append(train_loss / len(train_acc))
        test_loss_record.append(test_loss / len(test_acc))
        long_loss_record.append(long_loss / len(long_acc))

        train_record.append(trains_acc.cpu())
        test_record.append(tests_acc.cpu())
        long_record.append(longs_acc.cpu())
        print('--------------- Iteration', iteration + 1, '---------------')
        print('Train loss:', train_loss / len(train_acc))
        print('Test loss:', test_loss / len(test_acc))
        print('Train accuracy:', trains_acc)
        print('Test accuracy:', tests_acc)
        print('Long sentence accuracy:', longs_acc)
    
    return train_loss_record, test_loss_record, long_loss_record, train_record, test_record, long_record


def NN_embedding_plot(random_embedding, glove_embedding, learning_rate, batch_size, iter_times):
    # 获得训练集和测试集的batch
    train_random = getBatch(random_embedding.train_matrix, random_embedding.train_y, batch_size)
    test_random = getBatch(random_embedding.test_matrix, random_embedding.test_y, batch_size)
    train_glove = getBatch(glove_embedding.train_matrix, glove_embedding.train_y, batch_size)
    test_glove = getBatch(glove_embedding.test_matrix, glove_embedding.test_y, batch_size)

    # 模型建立
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    random_rnn = myRNN(50, 50, random_embedding.words_length)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    random_cnn = myCNN(50, random_embedding.words_length, random_embedding.longest)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    glove_rnn = myRNN(50, 50, glove_embedding.words_length, weight=torch.tensor(glove_embedding.trained_words, dtype=torch.float))
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    glove_cnn = myCNN(50, glove_embedding.words_length, glove_embedding.longest, weight=torch.tensor(glove_embedding.trained_words, dtype=torch.float))

    # rnn+random
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    train_loss_random_rnn, test_loss_random_rnn, long_loss_random_rnn, train_random_rnn, test_random_rnn, long_random_rnn = \
        NN_embedding(random_rnn, train_random, test_random, learning_rate, iter_times)
    # cnn+random
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    train_loss_random_cnn, test_loss_random_cnn, long_loss_random_cnn, train_random_cnn, test_random_cnn, long_random_cnn = \
        NN_embedding(random_cnn, train_random, test_random, learning_rate, iter_times)
    # rnn+glove
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    train_loss_glove_rnn, test_loss_glove_rnn, long_loss_glove_rnn, train_glove_rnn, test_glove_rnn, long_glove_rnn = \
        NN_embedding(glove_rnn, train_glove, test_glove, learning_rate, iter_times)
    # cnn+glove
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    train_loss_glove_cnn, test_loss_glove_cnn, long_loss_glove_cnn, train_glove_cnn, test_glove_cnn, long_glove_cnn = \
        NN_embedding(glove_cnn, train_glove, test_glove, learning_rate, iter_times)

    # 画图
    x = list(range(1, iter_times+1))
    plt.subplot(2, 2, 1)
    plt.plot(x, train_loss_random_rnn, 'r--', label='RNN+random')
    plt.plot(x, train_loss_random_cnn, 'g--', label='CNN+random')
    plt.plot(x, train_loss_glove_rnn, 'b--', label='RNN+glove')
    plt.plot(x, train_loss_glove_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title('Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')


    plt.subplot(2, 2, 2)
    plt.plot(x, test_loss_random_rnn, 'r--', label='RNN+random')
    plt.plot(x, test_loss_random_cnn, 'g--', label='CNN+random')
    plt.plot(x, test_loss_glove_rnn, 'b--', label='RNN+glove')
    plt.plot(x, test_loss_glove_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title('Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(x, train_random_rnn, 'r--', label='RNN+random')
    plt.plot(x, train_random_cnn, 'g--', label='CNN+random')
    plt.plot(x, train_glove_rnn, 'b--', label='RNN+glove')
    plt.plot(x, train_glove_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title('Train Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    plt.subplot(2, 2, 4)
    plt.plot(x, test_random_rnn, 'r--', label='RNN+random')
    plt.plot(x, test_random_cnn, 'g--', label='CNN+random')
    plt.plot(x, test_glove_rnn, 'b--', label='RNN+glove')
    plt.plot(x, test_glove_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title('Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig('main_plot.png')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(x, long_loss_random_rnn, 'r--', label='RNN+random')
    plt.plot(x, long_loss_random_cnn, 'g--', label='CNN+random')
    plt.plot(x, long_loss_glove_rnn, 'b--', label='RNN+glove')
    plt.plot(x, long_loss_glove_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title('Long Sentence Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(x, long_random_rnn, 'r--', label='RNN+random')
    plt.plot(x, long_random_cnn, 'g--', label='CNN+random')
    plt.plot(x, long_glove_rnn, 'b--', label='RNN+glove')
    plt.plot(x, long_glove_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title('Long Sentence Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig('long_sentence_plot.png')
    plt.show()
