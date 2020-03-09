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
    W = tf.Variable(tf.random_normal([1, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.square(y - hypothesis))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000019104)
    train = optimizer.minimize(loss)

    n_steps = 30000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: one_yearX, y: one_yearY})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))
        #print(sess.run(y, feed_dict={y: one_yearY}))
        #print(sess.run(hypothesis, feed_dict={x: one_yearX}))

        pred_x = x_data[M:N]
        pred_x = np.reshape(pred_x, [-1, 1])
        pred_y = sess.run(hypothesis, feed_dict={x: x_data})

        plt.figure(0)
        plt.plot(x_data, y_data, '-r')
        plt.plot(x_data, pred_y, '-b')
        plt.plot(one_yearX, one_yearY, '-g')
        plt.show()
        pred_y2 = sess.run(hypothesis, feed_dict={x: [[500]]})
        print(pred_y2)
