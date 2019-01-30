import tensorflow as tf
import input_data


def train_use_big_model(p):
    mnist = input_data.read_data_sets('./', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob1, keep_prob2 = tf.placeholder(tf.float32), tf.placeholder(tf.float32)

    net = RecognizeMnist(x, y, keep_prob1, keep_prob2)

    optimizer = tf.train.AdamOptimizer(learning_rate=p.learning_rate).minimize(net.loss)

    # 结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(net.pred, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(0, 50):
            for step in range(0, p.steps):
                imgs, labels = mnist.train.next_batch(p.batch_size)
                imgs = imgs.reshape([p.batch_size, 28, 28, 1])
                data_dict = {x: imgs, y: labels, keep_prob1: 0.25, keep_prob2: 0.5}

                _, acc = sess.run([optimizer, accuracy], feed_dict=data_dict)

            if step % 20 == 0:
                print('[epoch {} step {}] accuracy = {}'.format(epoch, step, acc))


class RecognizeMnist():
    def __init__(self, x, y, keep_prob1, keep_prob2):
        WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
        self.x = x
        self.y = y

        w1 = tf.get_variable('w1', [3, 3, 1, 32], tf.float32, WEIGHT_INITIALIZER)
        self.x = tf.nn.relu(tf.nn.conv2d(self.x, w1, strides=[1, 1, 1, 1], padding='VALID'))

        w2 = tf.get_variable('w2', [3, 3, 32, 64], tf.float32, WEIGHT_INITIALIZER)
        self.x = tf.nn.relu(tf.nn.conv2d(self.x, w2, strides=[1, 1, 1, 1], padding='VALID'))

        self.x = tf.nn.max_pool(self.x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#       DROPOUT
        self.x = tf.nn.dropout(self.x, keep_prob=keep_prob1)

        w3 = tf.get_variable('w3', [3, 3, 64, 64], tf.float32, WEIGHT_INITIALIZER)
        self.x = tf.nn.relu(tf.nn.conv2d(self.x, w3, strides=[1, 1, 1, 1], padding='VALID'))

        w4 = tf.get_variable('w4', [3, 3, 64, 64], tf.float32, WEIGHT_INITIALIZER)
        self.x = tf.nn.relu(tf.nn.conv2d(self.x, w4, strides=[1, 1, 1, 1], padding='VALID'))

#       DROPOUT
        self.x = tf.nn.dropout(self.x, keep_prob=keep_prob1)
#       Flatten
        self.x = tf.layers.flatten(self.x)
#       Full connected layer
        self.x = tf.layers.dense(self.x, 81, activation='relu')
        self.x = tf.nn.dropout(self.x, keep_prob=keep_prob2)
        self.x = tf.layers.dense(self.x, 10)

        self.pred = tf.nn.softmax(self.x)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.x))
