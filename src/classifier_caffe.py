from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys
from pathlib import Path

import caffe
import cv2
import numpy as np
from sklearn.svm import SVC

import facenet

FACE_FEED_SIZE = 160
HOME = str(Path.home())


class Model:
    def __init__(self, embedding_size=512,
                 model_path=os.path.join(HOME, 'workspace', 'ml-facenet-jetson', 'src', 'resnet_models')):
        caffe_prototxt = os.path.join(model_path, 'resnetInception-512.prototxt')
        if embedding_size == 128:
            caffe_prototxt = os.path.join(model_path, 'resnetInception-128.prototxt')
        caffe_model = os.path.join(model_path, 'inception_resnet_v1_conv1x1.caffemodel')
        self.net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)

    @staticmethod
    def norm_l2_vector(bottle_neck):
        sum = 0
        for v in bottle_neck:
            sum += np.power(v, 2)
        sqrt = np.max([np.sqrt(sum), 0.0000000001])
        vector = np.zeros(bottle_neck.shape)
        for (i, v) in enumerate(bottle_neck):
            vector[i] = v / sqrt
        return vector.astype(np.float32)

    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        print(mean, std)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def get_embeddings(self, img_path):
        img = cv2.imread(img_path)
        prewhitened = self.prewhiten(img)[np.newaxis]
        inputCaffe = prewhitened.transpose((0, 3, 1, 2))  # [1,3,160,160]
        self.net.blobs['data'].data[...] = inputCaffe
        self.net.forward()
        vector = self.norm_l2_vector(self.net.blobs['flatten'].data.squeeze())
        print('Embedding size %s' % str(len(vector)))
        return vector


def main_caffe(args):
    model = Model()
    dataset = None
    if args.use_split_dataset:
        dataset_tmp = facenet.get_dataset(args.data_dir)
        train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                            args.nrof_train_images_per_class)

        if args.mode == 'TRAIN':
            dataset = train_set
        elif args.mode == 'CLASSIFY':
            dataset = test_set
    else:
        dataset = facenet.get_dataset(args.data_dir)

    for cls in dataset:
        assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

    paths, labels = facenet.get_image_paths_and_labels(dataset)
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    nrof_images = len(paths)
    emb_array = np.zeros((nrof_images, 512))

    for i in range(len(paths)):
        emb_array[i] = model.get_embeddings(paths[i])

    classifier_filename_exp = os.path.expanduser(args.classifier_filename)
    if args.mode == 'TRAIN':
        # Train classifier
        print('Training classifier')
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    elif args.mode == 'CLASSIFY':
        print('Testing classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


BASE_DIR = os.path.dirname(__file__)
ALIGNED_PICS = os.path.join(HOME, 'workspace', 'ml-facenet-jetson', 'src', 'lfw_aligned')
facenet_model_checkpoint = os.path.join(HOME, 'workspace', 'ml-facenet-jetson', 'src', '20180402-114759')
classifier_model = os.path.join(HOME, 'workspace', 'ml-facenet-jetson', 'src', '20180402-114759',
                                'caffe_classifier.pkl')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.', default=ALIGNED_PICS)
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default=facenet_model_checkpoint)
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.',
                        default=classifier_model)
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=3)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=2)
    parser.add_argument('--device', type=str, help='mac, linux or jetson')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args.device == 'mac':
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    main_caffe(args)
