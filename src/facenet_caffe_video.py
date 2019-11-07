import argparse
import sys
import time
import os

import caffe
import cv2

from mtcnn_caffe import mtcnn_caffe
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display

WINDOW_NAME = 'CaffeMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green


class CaffeMtcnn:
    def __init__(self, caffe_model_path='/home/pawan/workspace/ml-facenet-jetson/src/mtcnn_caffe'):
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
    parser.add_argument('--device', type=str, default='mac', help='mac or jetson')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[1]), int(bb[3]), int(bb[0]), int(bb[2])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)


def loop_and_detect(cam, mtcnn, minsize):
    full_scrn = False
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            show_faces(img, dets, landmarks)
            cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
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
    img = cv2.imread('/home/pawan/workspace/ml-facenet-jetson/src/test.png')
    boundingboxes, points = mtcnn.detect(img)

    for face in boundingboxes:
        print(int(face[1]), int(face[3]), int(face[0]), int(face[2]))


if __name__ == '__main__':
    # test_mtcnn_caffe()
    main()
