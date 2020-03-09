import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

tf.set_random_seed(777)

def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


input_data_column_cnt = 5
output_data_column_cnt = 1

seq_length = 28
rnn_cell_hidden_dim = 20
forget_bias = 1.0
num_stacked_layers = 1
keep_prob = 1.0

num = 2000
learning_rate = 0.001

save_file = './ckpt/Bitcoin_3.ckpt'

raw_data = np.loadtxt('./csv/bitcoin.csv', dtype=np.str, delimiter=',')

price = raw_data[:, 0:-2]
price = np.array(price)
price = price.astype(np.float)

norm_price = min_max_scaling(price)

print("data[0] : ", price[0])
print("norm_data[0] : ", norm_price[0])
print("=" * 100)

volume = raw_data[:,-2:-1]
volume = np.array(volume)
volume = volume.astype(np.float)

norm_volume = min_max_scaling(volume)

print("volume[0] : ", volume[0])
print("norm_volume[0] : ", norm_volume[0])
print("=" * 100)

x = np.concatenate((norm_price,norm_volume),axis=1)
print("x[0] : ", x[0])

y = x[:, 1]
print("y[0] : ", y[0])
print("len(y) : ", len(y))

print("=" * 100)

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i : i + seq_length]
    _y = y[i + seq_length]
    #_x = np.array(_x, dtype=np.str)
    #_y = np.array(_y, dtype=np.str)
    dataX.append(_x)
    dataY.append(_y)
    if i==0:
        print(_x , "->", _y)

print("=" * 100)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

print("trainX.shape : ", trainX.shape)
print("trainY.shape : ", trainY.shape)

print("trainX[0][0][0] : ", trainX[0][0][0], " ", type(trainX[0][0][0]))
print("trainY[0] : ", trainY[0], " ", type(trainY[0]))

#trainX = np.reshape(trainX, [-1, train_size, input_data_column_cnt])
trainY = np.reshape(trainY, [-1, 1])

#print("=" * 100)
#print("trainX.shape after : ", trainX.shape)
#print("trainY.shape after : ", trainY.shape)

testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

testY = np.reshape(testY, [-1, 1])

print("testX.shape : ", testX.shape)
print("testY.shape : ", testY.shape)

X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X : ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y : ", Y)
print("=" * 100)

targets = tf.placeholder(tf.float32, [None, 1])
print("targets : ", targets)
predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions : ", predictions)
print("=" * 100)


# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)
print("=" * 100)

# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
# 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

loss = tf.reduce_sum(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))

train_errors = []
test_errors = []
test_predict = ""

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("trainY.shape after : ", trainY.shape)

saver = tf.train.Saver()

try:
    saver.restore(sess, save_file)
except:
    pass

test_error_prev = 987654321
for i in range(num):
    _, _loss = sess.run([train, loss], feed_dict={X : trainX, Y : trainY})
    if (i % 100 == 0 or i == num - 1):
        train_predict = sess.run(hypothesis, feed_dict={X : trainX})
        train_error = sess.run(rmse, feed_dict={targets : trainY, predictions : train_predict})
        train_errors.append(train_error)

        test_predict = sess.run(hypothesis, feed_dict={X : testX})
        test_error = sess.run(rmse, feed_dict={targets : testY, predictions : test_predict})
        test_errors.append(test_error)

        if(test_error_prev > test_error):
            saver.save(sess, save_file)
            test_error_prev = test_error

        print(i, "train_error : ", train_error, "test_error : ", test_error, "test-train : ", test_error - train_error)


plt.figure(1)
plt.plot(train_errors, '-r')
plt.plot(test_errors, '-b')
plt.xlabel('epoch')
plt.ylabel('rmse(root mean square error)')

plt.figure(2)
plt.plot(testY, '-r')
plt.plot(trainY, '-g')
plt.plot(test_predict, '-b')
plt.plot(train_predict, '-', color='gold')
plt.xlabel('Time')
plt.ylabel('Price')

plt.show()
