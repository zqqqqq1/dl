import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/MNIST_data",one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,[None,784])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32,[None , 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess.run(init)
for i in range(1000):
    batch_x ,batch_y = mnist.train.next_batch(100)
    sess.run([train_step,cross_entropy],feed_dict={x:batch_x,y_:batch_y})

correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
print(acc.eval({x:mnist.test.images,y_:mnist.test.labels}))