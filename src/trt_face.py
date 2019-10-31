# coding=utf-8
"""Face Detection and Recognition"""

import pickle
import os
import numpy as np
import tensorflow as tf
from scipy import misc

import detect_face
import facenet
from utils.mtcnn import TrtMtcnn

BASE_DIR = os.path.dirname(__file__)

gpu_memory_fraction = 0.3
# facenet_model_checkpoint = os.path.join(BASE_DIR, '20180402-114759')
# classifier_model = os.path.join(BASE_DIR, '20180402-114759', 'my_classifier.pkl')
facenet_model_checkpoint = '/home/pawan/20180408-102900'
# facenet_model_checkpoint = '/home/pawan/20180408-102900/frozen_graph.pb'
classifier_model = '/home/pawan/20180408-102900/my_classifier.pkl'
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self, device='mac'):
        self.detect = Detection(device=device)
        self.encoder = Encoder()
        self.identifier = Identifier()

    # def add_identity(self, image, person_name):
    #     faces = self.detect.find_faces(image)

    # if len(faces) == 1:
    #     face = faces[0]
    #     face.name = person_name
    #     face.embedding = self.encoder.generate_embedding(face)
    #     return faces

    def identify(self, image, device):
        faces = self.detect.find_faces(image)
        print("Found %s faces" % str(len(faces)))
        for face in faces:
            embedding = self.encoder.generate_embedding(face.image)
            print("Embeddings generated")
            name = self.identifier.identify(embedding)
            print("Found %s" % name)
            face.name = name
            # return embedding, name


class Identifier:
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        print("Predicting face")
        if face is not None:
            predictions = self.model.predict_proba([face])
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        prewhiten_face = facenet.prewhiten(face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # face_crop_margin = 32
    # face_crop_size = 160

    def __init__(self, face_crop_size=160, face_crop_margin=32, device='mac'):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self, device='mac'):

        if device == 'mac':
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    return detect_face.create_mtcnn(sess, None)
        else:
            self.mtcnn = TrtMtcnn()

    def find_faces(self, image, device):
        faces = []

        if device == 'mac':
            bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                        self.pnet, self.rnet, self.onet,
                                                        self.threshold, self.factor)
        else:
            bounding_boxes, landmarks = self.mtcnn.detect(minsize=self.minsize)

        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces

    # def __init__(self, face_crop_size=160, face_crop_margin=32):
    #     self.mtcnn = TrtMtcnn()

    # def _setup_mtcnn(self):
    #     with tf.Graph().as_default():
    #         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    #         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #         with sess.as_default():
    #             return align.detect_face.create_mtcnn(sess, None)

    # def find_faces(self, image):
    #     faces = []
    #
    #     dets, landmarks = self.mtcnn.detect(image, minsize=self.minsize)
    #     for bb, ll in zip(dets, landmarks):
    #         face = Face()
    #         face.container_image = image
    #         face.bounding_box = np.zeros(4, dtype=np.int32)
    #
    #         img_size = np.asarray(image.shape)[0:2]
    #         face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
    #         face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
    #         face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
    #         face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
    #         cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
    #         face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
    #
    #         faces.append(face)
    #
    #     return faces
