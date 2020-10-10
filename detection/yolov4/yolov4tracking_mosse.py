from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import tensorflow as tf
from os.path import join, dirname, realpath
import cv2
from tracking.yolov4.detection.algorithm import detection_stage
from tracking.yolov4.detection.algorithm_yolov4_tiny import detection_stage_yolov4_tiny
from tracking.yolov4.tracking import preprocessing
from tracking.yolov4.tracking import nn_matching
from tracking.yolov4.tracking.detection import Detection
from tracking.yolov4.tracking.tracker import Tracker
from tracking.yolov4.tools import generate_detections as gdet

from tracking.base import BaseCF

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''GPU Memory Allocation:
There are 2 alternative methods. In the first, resource assignment is done automatically.
In the second, we make limitations.
'''

''' First Method
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
set_session(sess)
'''

'''Second Method'''

def tf_no_warning():

    try:
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except ImportError:
        pass

tf_no_warning()


class YOLOV4(BaseCF):
    def __init__(self, detection_model_degree=2, detection_image_size=608, detection_score=0.2, interp_factor=0.125, sigma=2.):
        super(YOLOV4).__init__()
        self.interp_factor = interp_factor
        self.sigma = sigma
        if detection_model_degree == 0:
            opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))
        elif detection_model_degree == 1:
            opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))
        elif detection_model_degree == 2:
            opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.02)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=opts))
        self.detection_model = detection_model_degree
        self.detection_imagesize = detection_image_size
        self.detection_score = detection_score

    def init(self, first_frame, bbox):
        max_cosine_distance = 0.3
        nn_budget = None
        self.nms_max_overlap = 1.0
        self.detection_model = None

        if detection_stage == 0 or self.detection_model == 1:
            self.detection_model = detection_stage(self.detection_model, self.detection_score)
        else:
            self.detection_model = detection_stage_yolov4_tiny(self.detection_imagesize, self.detection_score)

        # deep_sort_parameters
        model_filename = join(join(dirname(realpath(__file__)), 'model_data_new'), 'mars-small128.pb')
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        if len(first_frame.shape) == 2:
            #assert first_frame.shape[2] == 3
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)

        first_frame = self.get_feature(first_frame)
        first_frame = first_frame.astype(np.float32)/255

        x, y, w, h = tuple(bbox)
        self._center = (x+w/2, y+h/2)
        self.w, self.h = w, h
        w, h = int(round(w)), int(round(h))
        self.cos_window = cos_window((w, h))
        self._fi = cv2.getRectSubPix(first_frame, (w, h), self._center)
        self._G = np.fft.fft2(gaussian2d_labels((w, h), self.sigma))
        self.crop_size = (w, h)
        self._Ai = np.zeros_like(self._G)
        self._Bi = np.zeros_like(self._G)
        for _ in range(8):
            fi = self._rand_warp(self._fi)
            Fi = np.fft.fft2(self._preprocessing(fi, self.cos_window))
            self._Ai += self._G*np.conj(Fi)
            self._Bi += Fi*np.conj(Fi)

    def update(self, current_frame, vis=False):
        try:
            image = Image.fromarray(current_frame[..., ::-1])
            boxs = self.detection_model.detect_image(image)
            features = self.encoder(current_frame, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            boxes = np.array([d.tlwh for d in detections])# Run NMS(non-maxima suppression) for better detection
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            self.tracker.predict()
            self.tracker.update(detections)
            #boxes = []

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    #continue
                    print('continue')
                bbox = track.to_tlbr()
                return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        except Exception as ex:
            print('Error in YOLOV4 : {}'.format(str(ex)))

    def update(self, current_frame, vis=False):
        if len(current_frame.shape) == 2:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

        current_frame = self.get_feature(current_frame)
        current_frame = current_frame.astype(np.float32)/255
        Hi = self._Ai / self._Bi
        fi = cv2.getRectSubPix(current_frame, (int(round(self.w)), int(round(self.h))), self._center)
        fi = self._preprocessing(fi, self.cos_window)
        Gi = Hi*np.fft.fft2(fi)
        gi = np.real(np.fft.ifft2(Gi))
        if vis is True:
            self.score = gi
        curr = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
        dy, dx = curr[0]-(self.h/2), curr[1]-(self.w/2)
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (x_c, y_c)
        fi = cv2.getRectSubPix(current_frame, (int(round(self.w)), int(round(self.h))), self._center)
        fi = self._preprocessing(fi, self.cos_window)
        Fi = np.fft.fft2(fi)
        self._Ai = self.interp_factor*(self._G*np.conj(Fi))+(1-self.interp_factor)*self._Ai
        self._Bi = self.interp_factor*(Fi*np.conj(Fi))+(1-self.interp_factor)*self._Bi
        return [self._center[0]-self.w/2, self._center[1]-self.h/2, self.w, self.h]

    def _preprocessing(self, img, cos_window, eps=1e-5):
        img=np.log(img+1)
        img=(img-np.mean(img))/(np.std(img)+eps)
        return cos_window*img

    def _rand_warp(self, img):
        h, w = img.shape[:2]
        C = .1
        ang = np.random.uniform(-C, C)
        c, s = np.cos(ang), np.sin(ang)
        W = np.array([[c + np.random.uniform(-C, C), -s + np.random.uniform(-C, C), 0],
                      [s + np.random.uniform(-C, C), c + np.random.uniform(-C, C), 0]])
        center_warp = np.array([[w / 2], [h / 2]])
        tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
        W[:, 2:] = center_warp - center_warp * tmp
        warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
        return warped

    def get_feature(self):
        return self.tracker.tracks[0].features

