import tensorflow as tf

LEARNINGRATE = 1e-3


# 用来创建卷积层并把本层的参数存入参数列表
# input_op:输入的tensor name:该层的名称 kh:卷积层的高 kw:卷积层的宽 n_out:输出通道数，dh:步长的高 dw:步长的宽，p是参数列表
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    # 输入的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val , trainable=True , name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        return activation


# 定义全连接层
def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p += [kernel,biases]
        return activation


# 定义最大池化层
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)


# 定义网络结构
def inference(features, one_hot_labels):
    x = tf.pad(features, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
    with tf.variable_scope('reference') :
        p = []
        conv1_1 = conv_op(x,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
        conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
        pool1 = mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)

        conv2_1 = conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
        conv2_2 = conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
        pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

        conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

        conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

        conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

        shp = pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")

        keep_prob = tf.placeholder(tf.float32)
        fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)
        fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
        fc7 = fc_op(fc6_drop,name="fc7",n_out=4096,p=p)
        fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
        fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,p=p)

        y_conv = tf.layers.dense(fc8, units=61, activation=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cross_entropy)

    return train_step, cross_entropy, y_conv, keep_prob
