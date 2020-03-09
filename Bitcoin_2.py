import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt
tf = tf_new.compat.v1
tf.set_random_seed(777)

N = 397
M = 365
data = np.loadtxt('bitcoin.csv', dtype=np.str, delimiter=',')

DATE = []
for i in range(N):
    DATE.append(i)

one_yearX = DATE[0:M]

OPEN = data[:,2]
END = data[:,1]

OPEN = np.array(OPEN, dtype=np.float)
END = np.array(END, dtype=np.float)

AVERAGE = []
for i in range(N):
    AVERAGE.append((OPEN[i] + END[i]) / 2.0)

one_yearY = AVERAGE[0:M]

x_data = np.array(DATE)
y_data = np.array(AVERAGE)
one_yearX = np.array(one_yearX)
one_yearY = np.array(one_yearY)

x_data = np.reshape(x_data, [-1, 1])
y_data = np.reshape(y_data, [-1, 1])
one_yearX = np.reshape(one_yearX, [-1, 1])
one_yearY = np.reshape(one_yearY, [-1, 1])

g = tf.Graph()
with g.as_default() as graph:
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    W1 = tf.Variable(tf.random_normal([1, 2]))
    b1 = tf.Variable(tf.random_normal([2]))
    L1 = tf.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([2, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    L2 = tf.matmul(L1, W2) + b2

    hypothesis = L2

    loss = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.06)
    train = optimizer.minimize(loss)

    n_steps = 60000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: one_yearX, y: one_yearY})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))

        pred_x = x_data[M:N]
        pred_x = np.reshape(pred_x, [-1, 1])
        pred_y = sess.run(tf.round(hypothesis), feed_dict={x: x_data})

        plt.figure(0)
        plt.plot(x_data, y_data, '-r')
        plt.plot(x_data, pred_y, '.b')
        plt.plot(one_yearX, one_yearY, '-g')
        plt.show()
        correct_prediction = tf.equal(pred_y, y_data)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data, y: y_data}) * 100))
