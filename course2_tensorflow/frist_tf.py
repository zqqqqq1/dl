import tensorflow as tf
import h5py
import numpy as np
from course2_tensorflow import tf_utils
import matplotlib.pyplot as plt
X_train_orig , Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()


#对数据进行扁平化
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

#归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#换换为one_hot矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig , 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)

#我们的目标是构建一个能高准确率识别符号的算法
#目前的模型是：LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX，SIGMOID输出层已经转换为SOFTMAX。当有两个以上的类时，一个SOFTMAX层将SIGMOID一般化。


#创建Placeholder 占位符
def create_placeholder(n_x,n_y):
    """
    
    :param n_x: 图片向量的大小  64*64*3 = 12288 
    :param n_y: 分类树 0-5 = 6
    :return: 
        x: 一个数据输入的占位符 [n_x,None] dtype = "float"
        y: 一个对应的标签的占位符 [n_y,None] dtype = "float"
    """
    X = tf.placeholder(tf.float32,[n_x,None],name = "X")
    Y = tf.placeholder(tf.float32,[n_y,None],name = "Y")

    return X,Y

#test
# X , Y = create_placeholder(12288,6)
# print(X)
# print(Y)

#初始化参数
#初始化tensorflow中的参数，我们将使用Xavier初始化权重和用零来初始化偏差。比如


def initialize_parameters():
    """
    W1 : [25,12288]
    b1 ；[25,1]
    W2: [12,25]
    b2 : [12,1]
    W3: [ 6,12]
    b3； [6,1]
    
    :return: 
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25,1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {
        "W1": W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }
    return parameters

#test
# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))
#
#
#
#




#实现forward_propagate

def forward_propagation(X,parameters):
    """
    实现一个模型的前向传播,
    模型结构为：Linear -> relu - > linear -> relu -> linear ->softmax
    :param X:  输入数据的占位符
    :param parameters:  包含了w和b的参数的字典
    :return: 
        Z3: 最后一个节点的linear输出

    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.matmul(W1,X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2,A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3,A2) + b3

    return Z3




#计算cost
def compute_cost(Z3,Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels = labels))

    return cost

#得益于 编程框架，所有的反向传播和参数更新都在1行代码中间进行锤了
#计算cost之后，将创建一个optimizer 对象
#运行tf.seesion时，必须将此对象和成本函数一起调用，当被调用时，它将使用所选择的方法和学习速率对cost进行优化



def model(X_train,Y_train,X_test,Y_test
          ,learning_rate  = 0.0001,num_epochs = 1500,minibatch_size =32,
          print_cost = True,is_plot = True):
    """
    实现一个三层的tensorflow神经网络，linear - relu - linear - relu - linear - softmax
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param Y_test: 
    :param learning_rate: 
    :param num_epochs: 
    :param minibatch_size: 
    :param print_cost: 
    :param is_plot: 
    :return: 
    """
    tf.reset_default_graph()

    tf.set_random_seed(1)

    seed = 3
    (n_x, m ) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    #创建占位符
    X,Y  = create_placeholder(n_x,n_y)

    #初始化参数
    parameters = initialize_parameters()

    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    #反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    #开始会话并且计算
    with tf.Session() as sess:
        #初始化
        sess.run(init)

        #正常训练的循环
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed +1
            minibatchs =  tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatchs:
                (minibatch_X,minibatch_Y) = minibatch

                #数据已经准备好了，开始运行session
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})


                #计算误差
                epoch_cost = epoch_cost + minibatch_cost /num_minibatches

            #记录并且打印成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch)+ "   epoch_cost = "+ str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("iterations (per tens)")
            plt.title("Learning rate = " + str(learning_rate))
            plt.show()
        parameters = sess.run(parameters)
        print("参数已经保存到session中")

        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

        print("训练集的准确率:  " , accuracy.eval({X:X_train,Y:Y_train}))
        print("测试机的准确率:  " , accuracy.eval({X:X_test,Y:Y_test}))

        return parameters


import time
#开始时间
start_time = time.clock()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )

