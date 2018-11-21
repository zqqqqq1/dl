###实现双向lstm的简单模型
import	tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#我们读取mnist的数据
mnist = input_data.read_data_sets("data/",one_hot=True)
#然后设置一些训练参数
#学习率
learning_rate= 0.01 
#最大训练样本数
max_samples = 400000
#
batch_size = 128
#设置每间隔10词训练就展示一次训练情况
display_step = 10


#因为Mnist的图像尺寸为28*28，因此输入n_input为28
n_inputs = 28
#LSTM网络梯度的展开步数
n_steps = 28
#隐藏层单元
n_hidden = 256
#类别
n_classes = 10

#创建placeholder
x = tf.placeholder("float",[None,n_steps,n_inputs])
y = tf.placeholder("float",[None,n_classes])

weights = tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
bises = tf.Variable(tf.random_normal([n_classes]))

#下面定义BI_LSTM网络的生成函数
#数据处理
def BiRNN(x , weights, bises):
    #把形状为(batch_size , n_steps ,n_inputs)的输入x变为长度为n_steps的列表
    #而其中元素形状为(batch_size , n_inputs)

    #使用tf.transposes(x,[1,0,2]) 将x的第一个维度batch_size和第二个n_steps 交换
    x = tf.transpose(x,[1,0,2])
    #接着使用tf.reshape 将x变形为(n_steps*batch_size,n_inputs)
    x = tf.reshape(x,[-1,n_inputs])
    x = tf.split(x,n_steps)
    #列表中的每个tensor的尺寸都是(batch_size,n_inputs)，这样就符合了LSTM单元的输入格式

    #创建forward和backward的LSTM单元
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)


    #然后将正向和反向的LSTM单元传入Bi_RNN接口中，生成双向LSTM，并且传入x作为输入
    outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,
                                                          x,dtype = tf.float32)

    return tf.matmul(outputs[-1],weights)+bises

pred = BiRNN(x,weights,bises)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1) , tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x ,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size,n_steps,n_inputs))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter " +str(step*batch_size)+" , MiniBatch Loss = " + "{:.6f}".format(loss)+", Training Accracy= "
                  +"{:.5f}".format(acc))
        step += 1
    print("Optimization Finished")


    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_inputs))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy : ",
            sess.run(accuracy,feed_dict={x:test_data,y:test_label}))






















































