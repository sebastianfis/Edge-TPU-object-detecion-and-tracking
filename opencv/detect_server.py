# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 opencv/detect_server.py --model all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 opencv/detect_server.py --model all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels all_models/coco_labels.txt

"""
from flask import Response
from flask import Flask
from flask import render_template
import threading
import datetime
import time
import os
import cv2
import argparse
import logging
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects, Object
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from tracker import ObjectTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def run_server(interpreter, labels, args):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    # initialize the motion detector and the total number of frames
    inference_size = input_size(interpreter)
    cam = cv2.VideoCapture(args.camera_idx)
    if args.tracker is not None:
        mot_tracker = ObjectTracker(args.tracker).trackerObject.mot_tracker
    else:
        mot_tracker = None
    assert cam is not None
    while True:
        try:
            res, image = cam.read()
            if res is False:
                logger.error("Empty image received")
                break
            else:
                timestamp = time.time()
                cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
                run_inference(interpreter, cv2_im_rgb.tobytes())
                objs = get_objects(interpreter, args.threshold)[:args.top_k]
                trdata = []
                trackerFlag = False
                if mot_tracker is not None:
                    detections = []  # np.array([])
                    for n in range(0, len(objs)):
                        element = []  # np.array([])
                        element.append(objs[n].bbox.xmin)
                        element.append(objs[n].bbox.ymin)
                        element.append(objs[n].bbox.xmax)
                        element.append(objs[n].bbox.ymax)
                        element.append(objs[n].score)  # print('element= ',element)
                        detections.append(element)  # print('dets: ',dets)
                    # convert to numpy array #      print('npdets: ',dets)
                    detections = np.array(detections)
                    if detections.any():
                        trdata = mot_tracker.update(detections)
                        trackerFlag = True
                frame = append_objs_to_img(image, inference_size, objs, labels, trackerFlag, trdata)

                tinference = time.time() - timestamp
                logger.info("Frame done in {}".format(tinference))
                cv2.putText(image, datetime.datetime.now().strftime(
                    "%A %d %B %Y %I:%M:%S%p"), (10, image.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                with lock:
                    outputFrame = frame.copy()
        except KeyboardInterrupt:
            break
    cam.release()


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels_de.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default='0.0.0.0', help="ip address of the device")
    parser.add_argument("-o", "--port", type=int, default='4664', help="port number of the server (1024 to 65535)")
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=1)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='detector score threshold')
    parser.add_argument('--tracker', help='Name of the Object Tracker To be used.',
                        default=None,
                        choices=[None, 'sort'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    t = threading.Thread(target=run_server, args=(interpreter, labels, args))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args.ip, port=args.port, debug=True, threaded=True, use_reloader=False)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def append_objs_to_img(cv2_im, inference_size, objs, labels, trackerFlag, trdata):
    height, width, channels = cv2_im.shape
    inf_w, inf_h = inference_size
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    if trackerFlag and (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = td[0].item(), td[1].item(
            ), td[2].item(), td[3].item(), td[4].item()
            overlap = 0
            for ob in objs:
                dx0, dy0, dx1, dy1 = ob.bbox.xmin.item(), ob.bbox.ymin.item(
                ), ob.bbox.xmax.item(), ob.bbox.ymax.item()
                area = (min(dx1, x1) - max(dx0, x0)) * (min(dy1, y1) - max(dy0, y0))
                if (area > overlap):
                    overlap = area
                    obj = ob

            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {} ID:{}'.format(
                percent, labels.get(obj.id, obj.id), int(trackID))
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            logger.info(label)
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return cv2_im


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    main()
