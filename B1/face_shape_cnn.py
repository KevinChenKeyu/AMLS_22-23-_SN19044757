import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import cv2


# training parametres
learning_rate = 0.0001
num_steps = 2000
batch_size = 2048


# net work parametres
num_in = 250000  # resolution of pictures is 500*500
num_classes = 5  # 5 kinds of face shape
dropout = 0.25  # Drop out, probability to dro a unit

def set_conv(x_dict, n_classes, dropout, reuse, istraining): # set up CNN

    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']

        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        layer_1 = tf.keras.layers.Conv2D(x, 32, 5, activation=tf.nn.relu)
        layer_1 = tf.keras.layers.MaxPooling2D(layer_1, 2, 2)

        layer_2 = tf.keras.layers.Conv2D(layer_1, 64, 3, activation=tf.nn.relu)
        layer_2 = tf.keras.layers.MaxPooling2D(layer_2, 2, 2)


        fully_connected_layer = tf.compat.v1.layers.flatten(layer_2)

        fully_connected_layer = tf.keras.layers.Dense(fully_connected_layer, 1024)

        fully_connected_layer = tf.keras.layers.Dropout(fully_connected_layer, rate=dropout, training=istraining)

        output = tf.keras.layers.Dense(fully_connected_layer, n_classes)

    return output


def model_function(features, labels, mode):

    logits_tr = set_conv(features, num_classes, dropout, reuse=False, istraining=True)
    logits_te = set_conv(features, num_classes, dropout, reuse=True, istraining=False)


    predict_class = tf.compat.v1.argmax(logits_te, axis=1)
    predict_prob = tf.nn.softmax(logits_te)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predict_class)


    losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tr, labels=tf.cast(
        labels, dtype=tf.int32)))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train = optimizer.minimize(losses, global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict_class)


    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=predict_class,loss=losses, train_op=train,
                                             eval_metric_ops={'accuracy': accuracy})

    return estim_specs


model = tf.estimator.Estimator(model_function)

input_function = tf.estimator.inputs.numpy_input_fn()











