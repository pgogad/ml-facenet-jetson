import argparse
import sys
import time

import cv2

import trt_face
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display

WINDOW_NAME = 'TestWindow'
BBOX_COLOR = (0, 255, 0)


def parse_args():
    desc = 'Capture and display live camera video'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40, help='minsize (in pixels) for detection [40]')
    parser.add_argument('--device', type=str, default='mac', help='mac or jetson')
    args = parser.parse_args()
    return args


def show_faces(img, recognizer, device='mac'):
    recognizer.identify(img, device)


def detect_faces(cam, recognization, minsize=40, device='mac'):
    full_scrn = False
    frame_interval = 5  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.localtime().tm_sec

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            if (frame_count % frame_interval) == 0:
                show_faces(img, recognization, device)
                # Check our current fps
                end_time = time.localtime().tm_sec
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    print("FPS %s" % int(frame_rate))
                    start_time = time.time()
            else:
                frame_count += 1
            cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('F') or key == ord('f'):
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)

    recognizer = trt_face.Recognition(args.device)

    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    cam.start()
    open_window(WINDOW_NAME, width=640, height=480, title='MTCNN Window')
    detect_faces(cam, recognizer, args.device)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
