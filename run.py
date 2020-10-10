import os
from main_tracker import PyTracker
from lib.utils import get_ground_truthes, plot_precision, plot_success
from os.path import join, dirname, realpath

class OTBDatasetConfig:
    """
    David(300:770), Football1(1:74), Freeman3(1:460), Freeman4(1:283)
    """
    frames={
        "David": [300, 770],
        "Football1": [1, 74],
        "Freeman4": [1, 283],
        'Freeman3': [1, 460],
        'Biker': [1, 142]
    }

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

tracker_type = 'MOSSE' # YOLOV4 VGG16

def create_fig(imreadf, imwritef):
    #data_dir = 'D:/PROJECTS/pyCFTrackers-master/pyCFTracker-master/dataset/OTB50/Freeman4/img/RESULT/MOSSE'  # Elnura ../dataset/test
    data_names = sorted(os.listdir(imreadf))
    length = len(data_names)

    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 4
    for i in range(1, columns * rows + 1):
        # img = np.random.randint(10, size=(h,w))
        img0 = cv2.imread(imreadf + '/' + str(int(i + (i-1)* length / (columns * rows))) + '.jpg')
        img = imutils.resize(img0, int(img0.shape[0]), int(img0.shape[1]))
        fig.add_subplot(rows, columns, i)
        fig.axes[i - 1].xaxis.set_ticks([])
        fig.axes[i - 1].yaxis.set_ticks([])
        plt.imshow(img)

    # plt.show()
    # fig.canvas.set_window_title('Test')
    plt.savefig(imwritef)

if __name__ == '__main__':
    data_dir = join(join(dirname(realpath(__file__)), 'DATASET'), 'OTB50')
    data_names = sorted(os.listdir(data_dir))

    print(data_names)
    dataset_config = OTBDatasetConfig()

    for data_name in data_names:
        print(data_name)
        data_path = join(data_dir, data_name)
        gts = get_ground_truthes(data_path)
        if data_name in dataset_config.frames.keys():
            start_frame, end_frame = dataset_config.frames[data_name][:2]
            if data_name != 'David':
                gts = gts[start_frame - 1:end_frame]
        img_dir = os.path.join(data_path, 'img')  # Elnura color names (CN)
        #trackers___ = ['MOSSE', 'KCF_GRAY', 'CN']
        tracker = PyTracker(img_dir, tracker_type=tracker_type, dataset_config=dataset_config)  # DCF_HOG DCF_GRAY STRCF ECO KCF_GRAY MOSSE KCF_CN CN ECO-HC CSRDCF

        video_path = join(join(data_dir, data_name), 'img/RESULT/' + tracker_type + '_vis.avi')
        success_plot_path = join(join(data_dir, data_name), 'img/RESULT/' + tracker_type + '_success.png')
        precision_plot_path = join(join(data_dir, data_name), 'img/RESULT/' + tracker_type + '_precision.png')

        poses = tracker.tracking(verbose=True, video_path=video_path)
        plot_success(gts, poses, success_plot_path)
        plot_precision(gts, poses, precision_plot_path)

    '''
        for data_name in data_names:
        print(data_name)
        data_path = os.path.join(data_dir, data_name) # Elnura

        img_dir1 = os.path.join(data_path,'img/RESULT/MOSSE')
        img_dir2 = os.path.join(data_path, 'img/RESULT/KCF_CN')
        img_dir3 = os.path.join(data_path, 'img/RESULT/KCF_GRAY')

        create_fig(img_dir1, 'MOSSE ' + data_name +'.png')
        create_fig(img_dir2, 'KCF_CN ' + data_name +'.png')
        create_fig(img_dir3, 'KCF_GRAY ' + data_name +'.png')
    '''






