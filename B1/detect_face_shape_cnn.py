import tensorflow as tf
import numpy as np
import cv2
from keras import Model, layers
from B1 import extract_data_cnn as data_required

num_class = 5 # 5 kind of face shape

learning_rate = 0.001
training_steps = 500
batch_size = 128
display_step = 10

filter1 = 32
filter2 = 64
fully_connect_1 = 1024

#from keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
#x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
#x_train, x_test = x_train / 255., x_test / 255.
# Use tf.data API to shuffle and batch data.
#train_img = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_img = train_img.repeat().shuffle(5000).batch(batch_size).prefetch(1)
X, Y = data_required.get_train_data()

train_img = tf.data.Dataset.from_tensor_slices((X, Y))
train_img = train_img.repeat().batch(batch_size).prefetch(1)


class CONVOLUTION(Model):
    def __init__(self):
        super(CONVOLUTION, self).__init__()
        channel = 3
        self.conv1 = layers.Conv2D(16, kernel_size=3, activation=tf.nn.relu, padding="valid", input_shape=(500, 500, 3))
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        self.conv2 = layers.Conv2D(16, kernel_size=3, activation=tf.nn.relu, padding="valid")
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        self.conv3 = layers.Conv2D(16, kernel_size=3, activation=tf.nn.relu, padding="valid")
        self.maxpool3 = layers.MaxPool2D(2, strides=2)

        self.conv4 = layers.Conv2D(16, kernel_size=3, activation=tf.nn.relu, padding="valid")
        self.maxpool4 = layers.MaxPool2D(2, strides=2)

        self.conv5 = layers.Conv2D(16, kernel_size=3, activation=tf.nn.relu, padding="valid")
        self.maxpool5 = layers.MaxPool2D(2, strides=2)

        self.flatten = layers.Flatten()

        self.fully_connect1 = layers.Dense(1024)

        self.dropout = layers.Dropout(rate=0.5)
        self.out = layers.Dense(num_class)


    def call(self, x, is_training=False, mask=None):
        x = tf.reshape(x, [-1, 500, 500, 3])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fully_connect1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)

        return x

convolution_net = CONVOLUTION()


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def get_accuracy(pred_y, correct_y):
    correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.cast(correct_y, tf.int64))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=-1)


optimizer = tf.optimizers.Adam(learning_rate)

def start_optimize(x,y):

    with tf.GradientTape() as i:
        predict = convolution_net(x, is_training=True)

        loss = cross_entropy_loss(predict, y)

    variable_tr = convolution_net.trainable_variables

    grad = i.gradient(loss, variable_tr)
    optimizer.apply_gradients(zip(grad, variable_tr))


for step, (x_batch, y_batch) in enumerate (train_img.take(training_steps), 1):

    start_optimize(x_batch, y_batch)

    if step % display_step == 0:
        pred = convolution_net(x_batch)
        loss = cross_entropy_loss(pred, y_batch)
        accuracy = get_accuracy(pred, y_batch)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, accuracy))







