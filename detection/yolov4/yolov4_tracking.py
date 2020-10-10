from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import tensorflow as tf
from os.path import join, dirname, realpath

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
    def __init__(self, detection_model_degree=2, detection_image_size=608, detection_score=0.2):
        super(YOLOV4).__init__()
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
                #boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                #(x, y) = (int(bbox[0]), int(bbox[1]))
                #(w, h) = (int(bbox[2]), int(bbox[3]))
                return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        except Exception as ex:
            print('Error in YOLOV4 : {}'.format(str(ex)))

