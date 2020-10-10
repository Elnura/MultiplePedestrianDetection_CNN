import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real
import argparse
import cv2
import os
from os.path import join, realpath, dirname
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from os.path import realpath, dirname, join
import cv2

model = VGG19()
ixs = [2, 5, 9, 13, 17]  # VGG16
ixs = [2, 5, 10, 15, 20]  # VGG19
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding=(0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)

    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h * grid, w * grid))
        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(np.pi / 9 * k))
                    cv2.rectangle(img, (j * grid, i * grid), ((j + 1) * grid, (i + 1) * grid), (255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.imshow("img", img)
        cv2.waitKey(0)


class Tracker():
    def __init__(self):
        self.max_patch_size = 256
        #self.max_patch_size = 64
        self.padding = 2.0
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = True

    def init(self, image, roi):
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        x = self.get_feature(image, roi)
        #x = self.get_cnn_feature(image, roi)

        y = self.gaussian_peak(x.shape[2], x.shape[1])
        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        cx, cy, w, h = self.roi
        max_response = -1
        for scale in [0.95, 1.0, 1.05]: # 0.95, 1.0, 1.05
            roi = map(int, (cx, cy, w * scale, h * scale))

            z = self.get_feature(image, roi)
            #z = self.get_cnn_feature(image, roi)

            responses = self.detect(self.alphaf, self.x, z, self.sigma)
            height, width = responses.shape
            #if self.debug:
                #cv2.imshow("res", responses)
                #c = cv2.waitKey(1) & 0xFF
                #if c == 27 or c == ord('q'):
                    #break

            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        self.roi = (cx + dx, cy + dy, best_w, best_h)
        # update template
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return (cx - w // 2, cy - h // 2, w, h)

    def get_feature(self, image, roi):
        cx, cy, w, h = roi
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        sub_image = image[y:y + h, x:x + w, :]
        #cv2.imshow('sub', sub_image)
        #c = cv2.waitKey(1) & 0xFF
        resized_image = cv2.resize(sub_image, (self.pw, self.ph))
        #cv2.imshow('resized', resized_image)
        #c = cv2.waitKey(1) & 0xFF

        if self.gray_feature:
            feature = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw) / 255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_image)
            #if self.debug:
                #self.hog.show_hog(feature)

        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def get_cnn_feature(self, image, roi):
        cx, cy, w, h = roi
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        sub_image = image[y:y + h, x:x + w, :]
        cv2.imshow('sub', image)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            return

        print(sub_image.shape)
        resized_image = cv2.resize(sub_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('resized', resized_image)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            return
        #resized_image = cv2.resize(sub_image, (self.pw, self.ph))

        #resized_image = img_to_array(resized_image)
        img = expand_dims(resized_image, axis=0)

        img = preprocess_input(img)
        feature_maps = model.predict(img)
        ffs = []
        for fmap in feature_maps:
            ffs.append(cv2.resize(fmap[0, :, :, :], (224, 224), interpolation=cv2.INTER_LINEAR))
        f = feature_maps[0][0, :, :, :]

        fc, fh, fw = f.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        return f

    def gaussian_peak(self, w, h):
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def train(self, x, y, sigma, lambdar):
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, alphaf, x, z, sigma):
        k = self.kernel_correlation(x, z, sigma)
        return real(ifft2(self.alphaf * fft2(k)))

    def kernel_correlation(self, x1, x2, sigma):
        c = ifft2(np.sum(conj(fft2(x1)) * fft2(x2), axis=0))
        c = fftshift(c)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

if __name__ == '__main__':
    tracker = Tracker()
    #roi = (262, 94, 16, 26)
    #roi = (275, 137, 23, 26)
    roi = (163, 44, 47, 164) #(114, 37, 32, 140)  (163, 44, 47, 164)

    #roi = (91, 58, 5, 16)

    img_dir = 'D:/PROJECTS/DATASET/OTB100/Skater2/img/'
    frame_list = get_img_list(img_dir)
    frame_list.sort()

    current_frame = cv2.imread(frame_list[0])
    #current_frame = cv2.resize(current_frame, (224, 224))
    #current_frame = load_img(frame_list[0], target_size=(224, 224))

    tracker.init(current_frame, roi)

    for idx in range(len(frame_list)):
        #frame = load_img(frame_list[idx], target_size=(224, 224))
        frame = cv2.imread(frame_list[idx])
        #frame = cv2.resize(frame, (224, 224))

        x, y, w, h = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imwrite('D:/PROJECTS/DATASET/OTB50/Biker/img/RESULT/VGG_KCF/' + str(idx) + '.jpg', frame)


        cv2.imshow('tracking', frame)



        c = cv2.waitKey(1) & 0xFF

        if c == 27 or c == ord('q'):
            break

    cv2.destroyAllWindows()




'''

frame = Image.fromarray(frame)


img = load_img('D:/PROJECTS/MOT_CNN_DATAASSOCIATION/DATASET/OTB50/Biker/img/0001.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)

img = preprocess_input(img)

feature_maps = model.predict(img)

for fmap in feature_maps:
    fmap_ix = fmap_ix + 1
    ix = 1

    fmap[0, :, :, ix - 1]

'''
