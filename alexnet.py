import argparse
import sys
import pickle
import numpy as np

import cifar10_utils
import cifar100_utils

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

cifar10_dataset_folder_path = 'cifar-10-batches-py'
save_model_path = './image_classification'

class AlexNet:
    def __init__(self, dataset, learning_rate):
        self.dataset = dataset

        if dataset == "cifar10":
          self.num_classes = 10
        elif dataset == "cifar100":
          self.num_classes = 100

        self.learning_rate = learning_rate
        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name='label')

        self.logits = self.load_model()
        self.model = tf.identity(self.logits, name='logits')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label), name='cost')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam').minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

    def load_model(self):
        # 1st
        conv1 = conv2d(self.input, num_outputs=96,
                    kernel_size=[11,11], stride=4, padding="VALID",
                    activation_fn=tf.nn.relu)
        lrn1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001,beta=0.75)
        pool1 = max_pool2d(lrn1, kernel_size=[3,3], stride=2)

        # 2nd
        conv2 = conv2d(pool1, num_outputs=256,
                    kernel_size=[5,5], stride=1, padding="VALID",
                    biases_initializer=tf.ones_initializer(),
                    activation_fn=tf.nn.relu)
        lrn2 = tf.nn.local_response_normalization(conv2, bias=2, alpha=0.0001, beta=0.75)
        pool2 = max_pool2d(lrn2, kernel_size=[3,3], stride=2)

        #3rd
        conv3 = conv2d(pool2, num_outputs=384,
                    kernel_size=[3,3], stride=1, padding="VALID",
                    activation_fn=tf.nn.relu)

        #4th
        conv4 = conv2d(conv3, num_outputs=384,
                    kernel_size=[3,3], stride=1, padding="VALID",
                    biases_initializer=tf.ones_initializer(),
                    activation_fn=tf.nn.relu)

        #5th
        conv5 = conv2d(conv4, num_outputs=256,
                    kernel_size=[3,3], stride=1, padding="VALID",
                    biases_initializer=tf.ones_initializer(),
                    activation_fn=tf.nn.relu)
        pool5 = max_pool2d(conv5, kernel_size=[3,3], stride=2)

        #6th
        flat = flatten(pool5)
        fcl1 = fully_connected(flat, num_outputs=4096,
                                biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
        dr1 = tf.nn.dropout(fcl1, 0.5)

        #7th
        fcl2 = fully_connected(dr1, num_outputs=4096,
                                biases_initializer=tf.ones_initializer(), activation_fn=tf.nn.relu)
        dr2 = tf.nn.dropout(fcl2, 0.5)

        #output
        out = fully_connected(dr2, num_outputs=self.num_classes, activation_fn=None)
        return out

    def label_to_name():
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def test(self, image, save_model_path):
        resize_images = []
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)

            loaded_x = loaded_graph.get_tensor_by_name('input:0')
            loaded_y = loaded_graph.get_tensor_by_name('label:0')
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

            resize_image = skimage.transform.resize(image, (224, 224), mode='constant')
            resize_images.append(resize_image)

            predictions = sess.run(
                tf.nn.softmax(loaded_logits),
                feed_dict={loaded_x: tmpTestFeatures, loaded_y: random_test_labels})

            label_names = load_label_names()

            predictions_array = []
            pred_names = []

            for index, pred_value in enumerate(predictions[0]):
                tmp_pred_name = label_names[index]
                predictions_array.append({tmp_pred_name : pred_value})

            return predictions_array

    def train_from_ckpt(self, epochs, batch_size, valid_set, save_model_path):
        tmpValidFeatures, valid_labels = valid_set

        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)

            loaded_x = loaded_graph.get_tensor_by_name('input:0')
            loaded_y = loaded_graph.get_tensor_by_name('label:0')
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

            optimizer = loaded_graph.get_operation_by_name('adam')

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = 5

                if self.dataset == 'cifar10':
                    n_batches = 5

                    for batch_i in range(1, n_batches + 1):
                        self._train_cifar10(sess,
                                            loaded_x, loaded_y, loaded_optimizer, loaded_acc,
                                            epoch, batch_i, batch_size, valid_set)

                else:
                    self._train_cifar100(sess,
                                         loaded_x, loaded_y, loaded_optimizer, loaded_acc,
                                         epoch, batch_size, valid_set)

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)

    def _train_cifar10(self, sess,
                        input, label, optimizer, accuracy,
                        epoch, batch_i, batch_size, valid_set):
        tmpValidFeatures, valid_labels = valid_set

        for batch_features, batch_labels in cifar10_utils.load_preprocess_training_batch(batch_i, batch_size):
            _ = sess.run(optimizer,
                        feed_dict={input: batch_features,
                                   label: batch_labels})

        print('Epoch {:>2}, CIFAR-10 Batch {}: '.format(epoch + 1, batch_i), end='')

        # calculate the mean accuracy over all validation dataset
        valid_acc = 0
        for batch_valid_features, batch_valid_labels in cifar10_utils.batch_features_labels(tmpValidFeatures, valid_labels, batch_size):
            valid_acc += sess.run(accuracy,
                                feed_dict={input:batch_valid_features,
                                           label:batch_valid_labels})

        tmp_num = tmpValidFeatures.shape[0]/batch_size
        print('Validation Accuracy {:.6f}'.format(valid_acc/tmp_num))

    def _train_cifar100(self, sess,
                          input, label, optimizer, accuracy,
                          epoch, batch_size, valid_set):
        tmpValidFeatures, valid_labels = valid_set

        for batch_features, batch_labels in cifar100_utils.load_preprocess_training_batch(batch_size):
            _ = sess.run(optimizer,
                        feed_dict={input: batch_features,
                                   label: batch_labels})

        print('Epoch {:>2}, CIFAR-100 : '.format(epoch + 1), end='')

        # calculate the mean accuracy over all validation dataset
        valid_acc = 0
        for batch_valid_features, batch_valid_labels in cifar100_utils.batch_features_labels(tmpValidFeatures, valid_labels, batch_size):
            valid_acc += sess.run(accuracy,
                                feed_dict={input:batch_valid_features,
                                            label:batch_valid_labels})

        tmp_num = tmpValidFeatures.shape[0]/batch_size
        print('Validation Accuracy {:.6f}'.format(valid_acc/tmp_num))

    def train(self, epochs, batch_size, valid_set, save_model_path):
        tmpValidFeatures, valid_labels = valid_set

        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())

            print('starting training ... ')
            for epoch in range(epochs):
                n_batches = 5

                if self.dataset == 'cifar10':
                    for batch_i in range(1, n_batches + 1):
                        self._train_cifar10(sess,
                                            self.input, self.label, self.optimizer, self.accuracy,
                                            epoch, batch_i, batch_size, valid_set)
                elif self.dataset == 'cifar100':
                    self._train_cifar100(sess,
                                         self.input, self.label, self.optimizer, self.accuracy,
                                         epoch, batch_size, valid_set)


            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running AlexNet')

    parser.add_argument('--dataset', help='imagenet or cifar10, cifar10 is the default', default='cifar10')
    parser.add_argument('--dataset-path', help='location where the dataset is present', default='none')
    parser.add_argument('--gpu-mode', help='single or multi', default='single')
    parser.add_argument('--learning-rate', help='learning rate', default=0.00005)
    parser.add_argument('--epochs', default=20)
    parser.add_argument('--batch-size', default=64)

    return parser.parse_args(args)

def main():
    args = sys.argv[1:]
    args = parse_args(args)

    dataset = args.dataset
    dataset_path = args.dataset_path
    gpu_mode = args.gpu_mode
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size

    if dataset == 'cifar10' and dataset_path == 'none':
        cifar10_utils.download(cifar10_dataset_folder_path)

    if dataset == 'cifar10':
        print('preprocess_and_save_data...')
        cifar10_utils.preprocess_and_save_data(cifar10_dataset_folder_path)

        print('load features and labels for valid dataset...')
        valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

        print('converting valid images to fit into imagenet size...')
        tmpValidFeatures = cifar10_utils.convert_to_imagenet_size(valid_features[:1000])
    else:
        sys.exit(0)

    alexNet = AlexNet(dataset, learning_rate)
    alexNet.train(epochs, batch_size, (tmpValidFeatures, valid_labels), save_model_path)

if __name__ == "__main__":
    main()





"""
def multi_gpu_convnet():
    # on GPU #1
    with tf.device('/gpu:0'):
        # 1st Convolutional Layer
        conv1_1 = tf.nn.conv2d(input, [11, 11, 3, 48], [1, 4, 4, 1], "SAME")
        lrn1_1 = tf.nn.local_response_normalization(conv1_1)
        pool1_1 = tf.nn.max_pool(lrn1_1)

        # 2nd Convolutional Layer
        conv2_1 = tf.nn.conv2d(pool1_1, [5, 5, 48, 128], [1, 4, 4, 1], "SAME")
        lrn2_1 = tf.nn.local_response_normalization(conv2_1)
        pool2_1 = tf.nn.max_pool(lrn2_1)

        conv3_1 = tf.nn.conv2d(pool2_1, [3, 3, 128, 192], [1, 4, 4, 1], "SAME")


    # on GPU #2
    with tf.device('/gpu:1'):
        # 1st Convolutional Layer
        conv1_2 = tf.nn.conv2d(input, [11, 11, 3, 48], [1, 4, 4, 1], "SAME")
        lrn1_2 = tf.nn.local_response_normalization(conv1_2)
        pool1_2 = tf.nn.max_pool(lrn1_2)

        # 2nd Convolutional Layer
        conv2_2 = tf.nn.conv2d(pool1_2, [5, 5, 48, 128], [1, 4, 4, 1], "SAME")
        lrn2_2 = tf.nn.local_response_normalization(conv2_2)
        pool2_2 = tf.nn.max_pool(lrn2_2)

        conv3_2 = tf.nn.conv2d(pool2_2, [3, 3, 128, 192], [1, 4, 4, 1], "SAME")

    ############ 3rd Convolutional Layer #########################################
    ##############################################################################
    ############ 4th Convolutional Layer #########################################

    with tf.device('/gpu:0'):
        conv4_1_input = tf.concat([conv3_1, conv3_2], 0)
        conv4_1 = tf.nn.conv2d(conv4_1_input, [3, 3, 192, 192], [1, 4, 4, 1], "SAME")

        conv5_1 = tf.nn.conv2d(conv4_1, [5, 5, 192, 128], [1, 4, 4, 1], "SAME")
        lrn5_1 = tf.nn.local_response_normalization(conv5_1)
        pool5_1 = tf.nn.max_pool(lrn5_1)

        flat_1 = tf.contrib.layers.flatten(pool5_1)

    with tf.device('/gpu:1'):
        conv4_2_input = tf.concat([conv3_2, conv3_1], 0)
        conv4_2 = tf.nn.conv2d(conv4_2_input, [3, 3, 192, 192], [1, 4, 4, 1], "SAME")

        conv5_2 = tf.nn.conv2d(conv4_2, [5, 5, 192, 128], [1, 4, 4, 1], "SAME")
        lrn5_2 = tf.nn.local_response_normalization(conv5_2)
        pool5_2 = tf.nn.max_pool(lrn5_2)

        flat_2 = tf.contrib.layers.flatten(pool5_2)

    with tf.device('/gpu:0'):
        fcl1_1_input = tf.concat([flat_1, flat_2], 0)
        fcl1_1 = tf.contrib.layers.fully_connected(fcl1_1_input, 2048)

    with tf.device('/gpu:1'):
        fcl1_2_input = tf.concat([flat_1, flat_2], 0)
        fcl1_2 = tf.contrib.layers.fully_connected(fcl1_2_input, 2048)

    with tf.device('/gpu:0'):
        fcl2_1_input = tf.concat([fcl1_1, fcl1_2], 0)
        fcl2_1 = tf.contrib.layers.fully_connected(fcl2_1_input, 2048)

    with tf.device('/gpu:1'):
        fcl2_2_input = tf.concat([fcl1_1, fcl1_2], 0)
        fcl2_2 = tf.contrib.layers.fully_connected(fcl2_2_input, 2048)

    with tf.device('/gpu:0'):
        fcl3_1_input = tf.concat([fcl2_1, fcl2_2], 0)
        out = tf.contrib.layers.fully_connected(fcl3_1_input, 1000, activation_fn=None)

    return out
"""
