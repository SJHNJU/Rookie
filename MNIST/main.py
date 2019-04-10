import input_data
import argparse
import tensorflow as tf
# from model import RecognizeMnist, train_use_big_model
import matplotlib.pyplot as plt
import numpy as np


def train(p):
    mnist = input_data.read_data_sets('./', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    ans = tf.placeholder(tf.float32, [None, 10])
    y = tf.layers.dense(x, 81, activation='relu')
    z = tf.layers.dense(y, 10)
    pred = tf.nn.softmax(z)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ans, logits=z)

    train_step = tf.train.AdamOptimizer(learning_rate=0.008, beta2=0.99).minimize(loss)

#   结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(ans, 1), tf.argmax(pred, 1))  # argmax返回一维张量中最大的值所在的位置, 后面的1/0 表示按照哪个维度返回最大值。equal在相同的那个维度上是true，不同的维度上是false
#   求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast转换数据类型，true转换成1, false转换成0, reduce_Mean不加参数则归一成标量，即算平均

    saver = tf.train.Saver(max_to_keep=1)  # 始终保存当前训练的最新的模型（其他的删掉

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for j in range(0, 21):
            for i in range(0, p.steps):
                imgs, labels = mnist.train.next_batch(p.batch_size)
                _, acc = sess.run([train_step, accuracy], feed_dict={x: imgs, ans: labels})
                print('[epoch {} step {}] acc {}'.format(j, i, acc))

            saver.save(sess, 'ckpt/mnist.ckpt', global_step=j)  # 会在模型的名字最后加上当前的epoch


def test():
    mnist = input_data.read_data_sets('./', one_hot=True)
    x = tf.placeholder(tf.float32, [1, 784])
    y = tf.layers.dense(x, 81, activation='relu')
    z = tf.layers.dense(y, 10)
    pred = tf.nn.softmax(z)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './ckpt/mnist.ckpt-20')
        for idx in range(80, 100):
            img, label = mnist.train.images[idx], mnist.train.labels[idx]
            res = sess.run(pred, feed_dict={x: img.reshape(1, 784)})
            print('Prediction is {0}; Correct answer is {1}'.format(np.argmax(res), np.argmax(label)))

            plt.imshow(img.reshape(28, 28), 'gray')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=50)
    args = parser.parse_args()
    args.steps = 55000 // args.batch_size

    test()
