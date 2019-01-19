
import tensorflow as tf
import numpy as np
import time
import argparse
import cv2
import json
import os
import plant_input
import network
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMAGE_SIZE = 128
IMAGE_CHANNEL = 3
CHECKFILE = './checkpoint/model.ckpt'
LOGNAME = 'plant'


def train(train_dir, annotations, max_step, checkpoint_dir='./checkpoint/'):
    # train the model
    print("train************************")
    plant_data = plant_input.plant_data_fn(train_dir, annotations)
    features = tf.placeholder("float32", shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name="features")
    labels = tf.placeholder("float32", [None], name="labels")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=61)
    train_step, cross_entropy, logits, keep_prob = network.inference(features, one_hot_labels)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    costs = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')
        logger = plant_input.train_log(LOGNAME)

        for step in range(start_step, start_step + max_step):
            start_time = time.time()
            x, y = plant_data.next_batch(BATCH_SIZE, IMAGE_SIZE)
            sess.run(train_step, feed_dict={features: x, labels: y, keep_prob: 0.7})
            if step % 50 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: x, labels: y, keep_prob: 1})
                train_loss = sess.run(cross_entropy, feed_dict={features: x, labels: y, keep_prob: 1})
                costs.append(train_loss)
                duration = time.time() - start_time
                logger.info("step %d: training accuracy %g, loss is %g (%0.3f sec)" % (step, train_accuracy, train_loss, duration))
            if step % 1000 == 1:
                saver.save(sess, CHECKFILE, global_step=step)
                print('writing checkpoint at step %s' % step)

        #X_train, Y_train = plant_data.whole_batch(IMAGE_SIZE)
        print("gettting whole training examples-------------------------------")
        X_train, Y_train = plant_data.not_actual_whole_batch(151, IMAGE_SIZE)
        #train_accuracy = accuracy.eval({features: X_train, labels: Y_train})
        print("whole traininig accuracy-------------------------------")
        cor_prediction = sess.run(correct_prediction, feed_dict={features: X_train, labels: Y_train, keep_prob: 1})
        print("correct prediction:", cor_prediction)
        train_accuracy = sess.run(accuracy, feed_dict={features: X_train, labels: Y_train, keep_prob: 1})
        print("Train Accuracy:", train_accuracy)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations ')
        plt.title("Learning rate ")
        plt.show()


def test(test_dir, checkpoint_dir='./checkpoint/'):
    # predict the result
    print("test************************")
    test_images = os.listdir(test_dir)
    features = tf.placeholder("float32", shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name="features")
    labels = tf.placeholder("float32", [None], name="labels")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=61)
    train_step, cross_entropy, logits, keep_prob = network.inference(features, one_hot_labels)
    #values, indices = tf.nn.top_k(logits, 3)
    values, indices = tf.nn.top_k(logits, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            raise Exception('no checkpoint find')

        result = []
        for test_image in test_images:
            temp_dict = {}
            x = plant_input.img_resize(os.path.join(test_dir, test_image), IMAGE_SIZE)
            predictions = np.squeeze(sess.run(indices, feed_dict={features: np.expand_dims(x, axis=0), keep_prob: 1}), axis=0)
            temp_dict['image_id'] = test_image
            #temp_dict['label_id'] = predictions.tolist()
            temp_dict['label_id'] = int(predictions[0])
            result.append(temp_dict)
            #print('image %s is %d,%d,%d' % (test_image, predictions[0], predictions[1], predictions[2]))
        
        with open('submit.json', 'w') as f:
            json.dump(result, f)
            print('write result json, num is %d' % len(result))
       

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        help="""\
        determine train or test\
        """
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default='../training/images/',
        help="""\
        determine path of trian images\
        """
    )

    parser.add_argument(
        '--annotations',
        type=str,
        default='../training/AgriculturalDisease_train_annotations.json',
        help="""\
        annotations for train images\
        """
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../validation/images/',
        help="""\
        determine path of test images\
        """
    )
    parser.add_argument(
        '--max_step',
        type=int,
        default=151,
        help="""\
        determine maximum training step\
        """
    )

    FLAGS = parser.parse_args()
    print("##############")
    print(FLAGS.train_dir)
    if FLAGS.mode == 'train':
        train(FLAGS.train_dir, FLAGS. annotations, FLAGS.max_step)
    elif FLAGS.mode == 'test':
        test(FLAGS.test_dir)
    else:
        raise Exception('error mode')
print('done')
