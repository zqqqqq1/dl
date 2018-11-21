import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def linear_function():
    """
        实现一个线性功能：
            初始化W，类型为tensor的随机变量，维度为(4,3)
            初始化X，类型为tensor的随机变量，维度为(3,1)
            初始化b，类型为tensor的随机变量，维度为(4,1)
        返回：
            result - 运行了session后的结果，运行的是Y = WX + b 

        """
    np.random.seed(1)

    X = np.random.randn(3,1)
    W = np.random.randn(4,3)
    b = np.random.randn(4,1)



    #线性
    Y = tf.matmul(W,X)+b

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()

    return result
# print("result is :",str(linear_function()))

def sigmoid(z):
    x = tf.placeholder(tf.float32,name="x")

    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})
    return result
# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(12) = " + str(sigmoid(12))).



def one_hot_matrix(labels,C):
    C = tf.constant(C,name="C")
    one_hot_matrix = tf.one_hot(indices = labels,depth=C,axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)

    sess.close()
    return one_hot
# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels,C=4)
# print(str(one_hot))



def ones(shapes):
    ones = tf.ones(shapes)

    sess = tf.Session()

    ones = sess.run(ones)

    sess.close()
    return ones
#print(str(ones(3)))