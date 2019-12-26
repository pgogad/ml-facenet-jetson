import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import caffe
import cv2
import numpy as np
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display

WINDOW_NAME = 'CaffeMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green
FACE_FEED_SIZE = 160

HOME = os.path.join(str(Path.home()), 'workspace/ml-facenet-jetson')


class CaffeClasifier:
    def __init__(self, path=os.path.join(HOME, 'src', '20180402-114759', 'caffe_classifier.pkl')):
        self.classifier = ''
        # Classify images
        print('Loading classifier')
        with open(path, 'rb') as infile:
            (self.model, self.class_names) = pickle.load(infile)

    def predict(self, embedding):
        if embedding is None:
            return
        predictions = self.model.predict_proba(embedding)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))


class FacenetCaffe:
    def __init__(self, caffe_model=os.path.join(HOME, 'src/resnet_models/inception_resnet_v1_conv1x1.caffemodel'),
                 caffe_weights=os.path.join(HOME, 'src/resnet_models/resnetInception-512.prototxt')):
        self.net = caffe.Net(caffe_weights, caffe_model, caffe.TEST)

    def norm_l2_Vector(self, bottle_neck):
        sum = 0
        for v in bottle_neck:
            sum += np.power(v, 2)
        sqrt = np.max([np.sqrt(sum), 0.0000000001])
        vector = np.zeros(bottle_neck.shape)
        for (i, v) in enumerate(bottle_neck):
            vector[i] = v / sqrt
        return vector.astype(np.float32)

    def get_vector(self, input):
        self.net.blobs['data'].data[...] = input
        self.net.forward()
        vector = self.norm_l2_Vector(self.net.blobs['flatten'].data.squeeze())
        return vector


class CaffeYolo:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img):
        tic = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        toc = time.time()
        print('Time taken for detection %s' % str(toc - tic))
        return faces


def parse_args():
    desc = 'Capture and display live camera video, while doing real-time face detection with TrtMtcnn on Jetson Nano'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40, help='minsize (in pixels) for detection [40]')
    parser.add_argument('--device', type=str, default='mac', help='mac, jetson or linux')
    args = parser.parse_args()
    return args


def show_faces(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), BBOX_COLOR, 2, 8)
        cv2.circle(img, (int(x + w / 2), int(y + h / 2)), int((max(w, h) / 2)), BBOX_COLOR, 2)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    print(mean, std)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def get_embeddings(img, face_caffe, faces, embedding_size=512):
    if len(faces) < 1:
        return None
    vectors = np.zeros((len(faces), embedding_size))

    i = 0
    for (x, y, w, h) in faces:
        cropped = img[y:y + h, x:x + w, :]
        cropped = cv2.resize(cropped, (FACE_FEED_SIZE, FACE_FEED_SIZE))
        prewhitened = prewhiten(cropped)[np.newaxis]
        input_caffe = prewhitened.transpose((0, 3, 1, 2))  # [1,3,160,160]
        tic = time.time()
        vector = face_caffe.get_vector(input_caffe)
        toc = time.time()
        print('Time taken for getting the vector : %s' % str(toc - tic))
        vectors[i] = vector
        i += 1
    return vectors


def loop_and_detect(cam, mtcnn, minsize):
    full_scrn = False
    face_caffe = FacenetCaffe()
    classifier = CaffeClasifier()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            faces = mtcnn.detect(img)
            if len(faces) > 0:
                show_faces(img, faces)
                emb = get_embeddings(img, face_caffe, faces, embedding_size=512)
                classifier.predict(emb)
            cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            del face_caffe
            del classifier
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main(args):
    if args.device == 'mac':
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    mtcnn = CaffeYolo()

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height, 'Camera TensorRT MTCNN Demo for Jetson Nano')
    loop_and_detect(cam, mtcnn, args.minsize)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

    del mtcnn


if __name__ == '__main__':
    args = parse_args()
    main(args)
