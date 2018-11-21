import tensorflow as tf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data",one_hot=True)

learning_rating = 0.001
trainint_iters = 100000
batch_size = 128
display_step = 100

frame_size =28
n_input = 28

n_hidden_units = 128
n_classes = 10

#placeholder
x = tf.placeholder(tf.float32,[None,n_input*frame_size],name="input_x")
y = tf.placeholder(tf.float32,[None,n_classes] ,name = "expected_y")

w = tf.Variable(tf.truncated_normal(shape=[n_hidden_units,n_classes]))
b = tf.Variable(tf.truncated_normal(shape=[n_classes]))

def RNN_Model(x,weight,bias):
    x = tf.reshape(x,shape=[-1,frame_size,n_input])
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden_units)
    init_state = tf.zeros(shape=[batch_size,rnn_cell.state_size])

    ouput,state = tf.nn.dynamic_rnn(rnn_cell,x,dtype=tf.float32)

    return tf.nn.softmax(tf.matmul(ouput[:,-1,:],weight)+bias,1)

pred = RNN_Model(x,w,b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rating).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(trainint_iters):
        batch_x ,batch_y = mnist.train.next_batch(batch_size)
        _loss,_ = sess.run([cost,optimizer],feed_dict={x:batch_x,y:batch_y})
        train_acc = accuracy.eval(feed_dict={x:batch_x,y:batch_y})
        if i % display_step ==0:
            print("step %d cost %g  acc %g"%(i,_loss,train_acc))

    print("test acc ")
    print(accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))