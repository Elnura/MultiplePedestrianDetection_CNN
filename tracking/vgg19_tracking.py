from PIL import Image
from keras.applications.vgg19 import VGG19 as VGG19_KERAS
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

import numpy as np
import cv2
from lib.utils import gaussian2d_labels,cos_window
from .base import BaseCF

class VGG19(BaseCF):
    def __init__(self, interp_factor=0.125, sigma=2.):
        super(VGG19).__init__()
        self.vgg_model = VGG19_KERAS()
        self.interp_factor = interp_factor
        self.sigma = sigma

    def init(self, first_frame, bbox):
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

    def get_feature(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = Image.fromarray(frame)
        #frame = cv2.resize(frame, (224, 224))
        ixs = [2, 5, 9, 13, 17]
        outputs = [self.vgg_model.layers[i].output for i in ixs]
        model = Model(inputs=self.vgg_model.inputs, outputs=outputs)
        img = img_to_array(frame)
        img = expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature_maps = model.predict(img)

        return feature_maps[0][0, :, :, 0]
