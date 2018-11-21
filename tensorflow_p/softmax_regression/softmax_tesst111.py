import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data",one_hot=True)

x = tf.placeholder(tf.float32,[None,28*28])
w = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.placeholder(tf.float32,[None,10])
y_pred = tf.nn.softmax(tf.matmul(x,w)+b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

costs = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        batch_x ,batch_y = mnist.train.next_batch(100)
        sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        train_acc = accuracy.eval(feed_dict={x:batch_x,y:batch_y})
        if i %100 ==0:
            print("step %d   accuracy ： %g"% (i,train_acc))
        costs.append(cost)

    print(accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))
