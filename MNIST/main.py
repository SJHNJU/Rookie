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
    correct_prediction = tf.equal(tf.argmax(ans, 1), tf.argmax(pred, 1))  # argmax返回一维张量中最大的值所在的位置
#   求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for j in range(0, 21):
            for i in range(0, p.steps):
                imgs, labels = mnist.train.next_batch(p.batch_size)
                _, acc = sess.run([train_step, accuracy], feed_dict={x: imgs, ans: labels})
                print('[epoch {} step {}] acc {}'.format(j, i, acc))

            saver.save(sess, 'ckpt/mnist.ckpt', global_step=j)


def test():
    mnist = input_data.read_data_sets('./', one_hot=True)
    x = tf.placeholder(tf.float32, [1, 784])
    y = tf.layers.dense(x, 81, activation='relu')
    z = tf.layers.dense(y, 10)
    pred = tf.nn.softmax(z)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './ckpt/mnist.ckpt-19')
        for idx in range(100, 110):
            img, label = mnist.train.images[idx], mnist.train.labels[idx]
            res = sess.run(pred, feed_dict={x: img.reshape(1, 784)})
            print(np.argmax(res))
            print(label)

            plt.imshow(img.reshape(28, 28), 'gray')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=50)
    args = parser.parse_args()
    args.steps = 55000 // args.batch_size

    train(args)
