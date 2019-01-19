import tensorflow as tf
LEARNINGRATE = 1e-3

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

def identity_block( X_input, kernel_size, in_filter, out_filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3)
        X = tf.nn.relu(X)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3)
        X = tf.nn.relu(X)

        # third

        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3)

        # final step
        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result

def convolutional_block( X_input, kernel_size, in_filter,
                        out_filters, stage, block, stride=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input
        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3)
        X = tf.nn.relu(X)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3)
        X = tf.nn.relu(X)

        # third
        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3)

        # shortcut path
        W_shortcut = weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        # final
        add = tf.add(x_shortcut, X)
        add_result = tf.nn.relu(add)

    return add_result

def inference(features, one_hot_labels):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """
    x = tf.pad(features, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
    with tf.variable_scope('reference') :
        #training = tf.placeholder(tf.bool, name='training')

        #stage 1
        w_conv1 = weight_variable([7, 7, 3, 64])
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=3)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='VALID')
        #assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

        #stage 2
        x = convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', stride=1)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c')

        #stage 3
        x = convolutional_block(x, 3, 256, [128,128,512], 3, 'a')
        x = identity_block(x, 3, 512, [128,128,512], 3, 'b')
        x = identity_block(x, 3, 512, [128,128,512], 3, 'c')
        x = identity_block(x, 3, 512, [128,128,512], 3, 'd')

        #stage 4
        x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd')
        x = identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f')

        #stage 5
        x = convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a')
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b')
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c')

        x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')

        x = tf.layers.flatten(x)
        #x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x, keep_prob)

        y_conv = tf.layers.dense(x, units=61, activation=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cross_entropy)
    return train_step, cross_entropy, y_conv, keep_prob

# def inference(features, one_hot_labels):
#     # network structure
#     # conv1
#     W_conv1 = weight_variable([5, 5, 3, 64], stddev=1e-4)
#     b_conv1 = bias_variable([64])
#     h_conv1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1)
#     h_pool1 = max_pool_3x3(h_conv1)
#     # norm1
#     norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#     # conv2
#     W_conv2 = weight_variable([5, 5, 64, 64], stddev=1e-2)
#     b_conv2 = bias_variable([64])
#     h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
#     # norm2
#     norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#     h_pool2 = max_pool_3x3(norm2)
#
#     # conv3
#     W_conv3 = weight_variable([5, 5, 64, 64], stddev=1e-2)
#     b_conv3 = bias_variable([64])
#     h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#     h_pool3 = max_pool_3x3(h_conv3)
#
#     # fc1
#     W_fc1 = weight_variable([16 * 16 * 64, 128])
#     b_fc1 = bias_variable([128])
#     h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*64])
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#
#     # introduce dropout
#     keep_prob = tf.placeholder("float")
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#     # fc2
#     W_fc2 = weight_variable([128, 80])
#     b_fc2 = bias_variable([80])
#     y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#     # calculate loss
#     cross_entropy = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_conv))
#     train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cross_entropy)
#
#     return train_step, cross_entropy, y_conv, keep_prob
