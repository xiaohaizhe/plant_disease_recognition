#coding=utf-8

import tensorflow as tf 
import numpy as np 
import cv2
import os
import glob
import tensorflow.contrib.slim as slim
import inception_v3 as v3
from create_tf_record import *



def predict(models_path, image_dir, labels_filename, labels_nums, data_format):
    '''
    预测过程
    :param models_path:     训练模型保存的路径
    :param image_dir:       待预测图像路径
    :param labels_filename: 类别名称文件
    :param labels_nums:     label数
    :param data_format:
    :return: None
    '''
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    with slim.arg_scope(v3.inception_v3_arg_scope()):
        out, end_points = v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(models_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception('no checkpoint find')

    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]
        print "{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score)

        result = []
        temp_dict = {}
        temp_dict['image_id'] = image_path
        temp_dict['label_id'] = pre_label
        result.append(temp_dict)
        
        with open('submit.json', 'w') as f:
            json.dump(result, f)
            print('write result json, num is %d' % len(result))
    sess.close()


if __name__ == '__main__':
    class_nums = 61 #类别个数
    image_dir = '../plant_disease/11.14/ai_challenger_pdr2018_testa_20181023/AgriculturalDisease_testA/images'
    labels_filename = '../plant_disease/11.14/label_cn.txt' #类别名称纪录文件，cn为中文名，en为英文名称
    models_path = 'models/model.ckpt-10000'

    batch_size = 1  #
    resize_height = 299  # 指定存储图片高度
    resize_width = 299   # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    predict(models_path, image_dir, labels_filename, class_nums, data_format)
