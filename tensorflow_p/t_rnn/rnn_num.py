from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#载入本地数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

learning_rating = 0.001
trainint_iters = 100000
batch_size = 128

n_input = 28
n_step = 28

n_hidden_units = 128
n_classes = 10

#placeholder
x = tf.placeholder(tf.float32,[None,n_step,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

weights = {
    'in':tf.Variable(tf.random_normal([n_input,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}


biases = {
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def rnn(X,weight,biases):

    X = tf.reshape(X,[-1,n_input])
    X_in = tf.matmul(X,weights["in"]+biases["in"])
    X_in = tf.reshape(X_in,[-1,n_step,n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)

    init = lstm_cell.zero_state(batch_size,dtype=tf.float32)

    with tf.variable_scope('tf.nn.dynsmic_rnn'):
        outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,init,time_major=False)

    results = tf.matmul(states[1],weights['out']+biases['out'])

    return results
tf.Graph()

pred = rnn(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rating).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size<trainint_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size,n_step,n_input])
        sess.run([train_op],feed_dict={x:batch_x,y:batch_y,})
        if step %50==0:
            print(sess.run(accuracy, feed_dict={
                x: batch_x, y: batch_y,
            }))
            step += 1
