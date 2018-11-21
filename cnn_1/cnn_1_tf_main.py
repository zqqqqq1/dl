"""
cnn 的tensorflow实现
"""


import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.framework import ops

from cnn_1 import cnn_utils
X_train_orig , Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
np.random.seed(1)
# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))
# index = 6
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
#创建placeholder
def create_placeholders(n_H0,n_W0,n_C0,n_y):
    """
    为session创建占位符
    :param n_H0: 高度
    :param n_W0: 宽度
    :param n_C0: 信道数
    :param n_y: 分类数
    :return: 
        X 输入数据X的占位符 [None,n_H0,n_W0,n_C0] 类型为float32
        Y 输入数据Y 的占位符 [None,n_y]
    """

    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])

    return X,Y

#初始化参数
def initialize_parameters():
    """
    初始化权值矩阵，这了我们把权值矩阵硬编码
    W1 : [4, 4, 3, 8]
    W2 : [2, 2, 8, 16]

    :return: 
        包含了tensor类型的W1,W2字典
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {
        "W1":W1,
        "W2":W2
    }
    return parameters

#前向传播
def forward_propagation(X,parameters):
    """
    实现前向传播
    conv2d -> relu -> maxpool -> conv2d -> relu -> maxpool ->faltten ->fuulConnected
    :param X:  输入数据的placeholder
    :param parameters: 包含了W1 和W2 的字典
    :return: 
            Z3 最后一个Linear的输出
    """

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    #CONV2D 步伐 1 填充方式 same
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")

    #relu
    A1 = tf.nn.relu(Z1)

    #max pool 窗口大小f 8x8 步伐 8x8 填充方式 same
    P1 = tf.nn.max_pool(A1,ksize = [1,8,8,1],strides=[1,8,8,1],padding="SAME")


    #Conv2d 步伐 1 填充方式 SAME
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")

    #relu
    A2 = tf.nn.relu(Z2)

    #max pool 过滤器大小4x4 步伐4x4 填充方式 SAME
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

    #一维化上层的输出
    P = tf.contrib.layers.flatten(P2)

    #全连接层 使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn = None)

    return Z3
# tf.reset_default_graph()
# np.random.seed(1)
#
# with tf.Session() as sess_test:
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#
#     a = sess_test.run(Z3,{X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print("Z3 = " + str(a))
#
#     sess_test.close()


#计算cost
def compute_cost(Z3,Y):
    """
    计算cost
    :param Z3:正向传播的最后一个linear节点的输出 维度为6 分类数
    :param Y: 标签向量的placeholder
    :return: 
    cost
    """
    logits = Z3
    labels = Y
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels = labels))
    return cost

# tf.reset_default_graph()
#
# with tf.Session() as sess_test:
#     np.random.seed(1)
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     a = sess_test.run(cost,{X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
#     print("cost = " + str(a))
#
#     sess_test.close()

#构建模型
def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.009,
          num_epochs = 100,minibatch_size=64,print_cost = True,is_Plot = True):
    """
    使用tensorflow  实现三层的卷积神经网络
    conv2d -> relu -> maxpool -> conv2d -> relu -> maxpool ->flatten -> fullyConnected
    :param X_train: 训练数据[None,64,64,3]
    :param Y_train: 训练数据对应的标签[None,n_y=6]
    :param X_test:  测试数据
    :param Y_test:  测试数据对应的标签
    :param learning_rate: 学习率
    :param num_epochs: 遍历整个数据集的次数
    :param minibatch_size: 
    :param print_cost: 
    :param is_Plot: 
    :return: 
            train_accuracy 实数，训练集的准确度
            test_accuracy 实数，测试集的准确度
            parameters  学习之后的参数
    """


    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3

    (m, n_H0,n_W0,n_C0) = X_train.shape

    n_y = Y_train.shape[1]
    costs = []

    X ,Y = create_placeholders(n_H0=n_H0,n_W0=n_W0,n_C0=n_C0,n_y=n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #初始化参数
        sess.run(init)
        #开始遍历数据集
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)

            seed = seed + 1
            minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:
                #选择一个数据块
                (minibatch_X,minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,cost] , feed_dict={X:minibatch_X,Y:minibatch_Y})

                #累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches

            if print_cost:
                if epoch % 5 == 0:
                    print("当前是第  "+ str(epoch)+" 次迭代，成本值为 : "+ str(minibatch_cost))

            if epoch % 1 ==0 :
                costs.append(minibatch_cost)

        if is_Plot:
            plt.plot(np.squeeze(costs))
            plt.xlabel('iteration (per tens)')
            plt.ylabel('cost')
            plt.title('Learning rate = '+ str(learning_rate))
            plt.show()
        predict_op  = tf.arg_max(Z3 , 1)
        corrent_prediction  = tf.equal(predict_op , tf.argmax(Y,1))

        #计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction , "float"))
        print("corrent_prediction accuracy = "+ str(accuracy))

        train_accuracy = accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})
        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuracy))

        return (train_accuracy, test_accuracy, parameters)
_, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)


