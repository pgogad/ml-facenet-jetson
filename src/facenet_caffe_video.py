import argparse
import os
import sys
import time

import caffe
import cv2
import numpy as np

from mtcnn_caffe import mtcnn_caffe
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display

WINDOW_NAME = 'CaffeMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green
FACE_FEED_SIZE = 160

HOME = '/home/pawan/workspace'


class FacenetCaffe:
    def __init__(self, caffe_model=os.path.join(HOME,
                                                'ml-facenet-jetson/src/resnet_models/inception_resnet_v1_conv1x1.caffemodel'),
                 caffe_weights=os.path.join(HOME, 'ml-facenet-jetson/src/resnet_models/resnetInception-512.prototxt')):
        self.net = caffe.Net(caffe_weights, caffe_model, caffe.TEST)

    def normL2Vector(bottleNeck):
        sum = 0
        for v in bottleNeck:
            sum += np.power(v, 2)
        sqrt = np.max([np.sqrt(sum), 0.0000000001])
        vector = np.zeros((bottleNeck.shape))
        for (i, v) in enumerate(bottleNeck):
            vector[i] = v / sqrt
        return vector.astype(np.float32)

    def get_vector(self, input):
        self.net.blobs['data'].data[...] = input
        self.net.forward(data=np.asarray(input))
        print('{}'.format(self.net.blobs['flatten'].data.squeeze()))
        # vector = self.normL2Vector(self.net.blobs['flatten'].data.squeeze())
        return None


class CaffeMtcnn:
    def __init__(self, caffe_model_path=os.path.join(HOME, 'ml-facenet-jetson/src/mtcnn_caffe')):
        self.threshold = [0.8, 0.8, 0.6]
        self.factor = 0.709
        self.PNet = caffe.Net(os.path.join(caffe_model_path, "det1.prototxt"),
                              os.path.join(caffe_model_path, "det1.caffemodel"), caffe.TEST)
        self.RNet = caffe.Net(os.path.join(caffe_model_path, "det2.prototxt"),
                              os.path.join(caffe_model_path, "det2.caffemodel"), caffe.TEST)
        self.ONet = caffe.Net(os.path.join(caffe_model_path, "det3.prototxt"),
                              os.path.join(caffe_model_path, "det3.caffemodel"), caffe.TEST)

    def detect(self, img, minsize=40):
        img_matlab = img.copy()
        tmp = img_matlab[:, :, 2].copy()
        img_matlab[:, :, 2] = img_matlab[:, :, 0]
        img_matlab[:, :, 0] = tmp
        tic = time.time()
        boundingboxes, points = mtcnn_caffe.detect_face(img_matlab, minsize, self.PNet, self.RNet, self.ONet,
                                                        self.threshold, False, self.factor)
        toc = time.time()
        print('Time taken for detection %s' % str(toc - tic))
        return boundingboxes, points


def parse_args():
    desc = 'Capture and display live camera video, while doing real-time face detection with TrtMtcnn on Jetson Nano'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40, help='minsize (in pixels) for detection [40]')
    parser.add_argument('--device', type=str, default='mac', help='mac, jetson or linux')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    print(mean, std)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def get_embeddings(img, face_caffe, boxes, landmarks):
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cropped = img[y1:y2, x1:x2, :]
        cropped = cv2.resize(cropped, (FACE_FEED_SIZE, FACE_FEED_SIZE))
        prewhitened = prewhiten(cropped)[np.newaxis]
        input_caffe = prewhitened.transpose((0, 3, 1, 2))  # [1,3,160,160]
        tic = time.time()
        vector = face_caffe.get_vector(input_caffe)
        toc = time.time()
        print('Time taken for getting the vector : %s' % str(toc - tic))
        return vector


def loop_and_detect(cam, mtcnn, minsize):
    full_scrn = False
    face_caffe = FacenetCaffe()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            show_faces(img, dets, landmarks)
            print('Caffe Vector = {}'.format(get_embeddings(img, face_caffe, dets, landmarks)))
            cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            del face_caffe
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main(args):
    # args = parse_args()
    cam = Camera(args)

    if args.device == 'mac':
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)

    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    mtcnn = CaffeMtcnn()

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height, 'Camera TensorRT MTCNN Demo for Jetson Nano')
    loop_and_detect(cam, mtcnn, args.minsize)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

    del mtcnn


def test_mtcnn_caffe():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    mtcnn = CaffeMtcnn()
    img = cv2.imread(os.path.join(HOME, 'ml-facenet-jetson/src/test.png'))
    boundingboxes, points = mtcnn.detect(img)

    for face in boundingboxes:
        print(int(face[0]), int(face[1]), int(face[2]), int(face[3]))


if __name__ == '__main__':
    # test_mtcnn_caffe()
    args = parse_args()
    if args.device == 'mac':
        HOME = '/Users/pawan/workspace'
    elif args.device == 'linux':
        HOME = '/home/azureadmin/workspace'
    else:
        HOME = '/home/pawan/workspace'
    main(args)
