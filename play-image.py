from math import sin, cos, radians
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

import multiprocessing


import cv2

from sleap.gui.video import video_demo
from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.nn.datagen import generate_images, generate_confidence_maps, generate_pafs
from sleap.nn.inference import find_all_peaks, match_peaks_paf, match_peaks_paf_par
from sleap.util import usable_cpu_count

def rotate_pafs(pafs, theta):
    theta_rad = radians(-theta)

    # rotate where the paf vectors are located
    pafs = ski.transform.rotate(pafs, theta, order=0)

    # rotate the individual paf vectors
    X0 = pafs[..., 0::2]
    Y0 = pafs[..., 1::2]

    X1 = (X0 * cos(theta_rad)) - (Y0 * sin(theta_rad))
    Y1 = (X0 * sin(theta_rad)) + (Y0 * cos(theta_rad))

    pafs[..., 0::2] = X1
    pafs[..., 1::2] = Y1

    return pafs

def demo_confmaps(confmaps, video):
    win = QtVideoPlayer(video=video)
    win.setWindowTitle("confmaps")
    win.show()

    def plot_confmaps(parent, item_idx):
        frame_conf_map = ConfMapsPlot(confmaps[parent.frame_idx,...])
        win.view.scene.addItem(frame_conf_map)

    win.changedPlot.connect(plot_confmaps)
    win.plot()

def demo_pafs(pafs, video):
    win = QtVideoPlayer(video=video)
    win.setWindowTitle("pafs")
    win.show()

    def plot_fields(parent, i):

        frame_pafs = pafs[parent.frame_idx, ...]
        # frame_pafs = rotate_pafs(frame_pafs, theta)

        aff_fields_item = MultiQuiverPlot(frame_pafs, show=None, decimation=1)
        win.view.scene.addItem(aff_fields_item)

    win.changedPlot.connect(plot_fields)
    win.plot()

if __name__ == "__main__":
    # load some images
    

    data_path = "tests/data/json_format_v2/centered_pair_predictions.json"

    labels = Labels.load_json(data_path)

    # only use initial frame(s)
    labels.labeled_frames = labels.labeled_frames[:2]
    skeleton = labels.skeletons[0]

    imgs = generate_images(labels)


    from PySide2 import QtWidgets
    from sleap.io.video import Video
    from sleap.gui.video import QtVideoPlayer
    from sleap.gui.confmapsplot import ConfMapsPlot
    from sleap.gui.quiverplot import MultiQuiverPlot

    theta = 70
    theta_rad = radians(-theta)

    # imgs[0] = ski.transform.rotate(imgs[0], theta)

    import timeit

    # print(timeit.timeit('pafs = generate_pafs(labels)', number=1, globals=globals()))

    video = Video.from_numpy(imgs * 255)

    confmaps, pafs = generate_confidence_maps(labels), generate_pafs(labels)
    
    app = QtWidgets.QApplication([])
    # video_demo(labels)
    # demo_confmaps(confmaps, video)
    # demo_pafs(pafs, video)

    peaks, peak_vals = find_all_peaks(confmaps)

    print(f"cpus: {usable_cpu_count()}")

    pool = multiprocessing.Pool(processes=usable_cpu_count())

    lf = match_peaks_paf_par(peaks, peak_vals, pafs, skeleton, video, range(video.frames), pool=pool)
    labels = Labels(lf)

    video_demo(labels)

    app.exec_()

    # reps = 10
    # t = timeit.timeit('rotate_pafs(pafs[0], theta)', number=reps, globals=globals())
    # print(f"pafs total time: {t} s, time per: {t/reps} s")

    # t = timeit.timeit('ski.transform.rotate(imgs[0], theta)', number=reps, globals=globals())
    # print(f"image total time: {t} s, time per: {t/reps} s")

    # app = QtWidgets.QApplication([])
    # vid = Video.from_numpy(imgs * 255)

    # win = QtVideoPlayer(video=vid)
    # win.setWindowTitle("pafs")
    # win.show()

    # def plot_fields(parent, i):

    #     frame_pafs = pafs[parent.frame_idx, ...]
    #     frame_pafs = rotate_pafs(frame_pafs, theta)

    #     aff_fields_item = MultiQuiverPlot(frame_pafs, show=None, decimation=1)
    #     win.view.scene.addItem(aff_fields_item)

    # win.changedPlot.connect(plot_fields)
    # win.plot()

    # app.exec_()
    






    # img2 = ski.transform.rotate(img, 30, order=0)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2)
    # ax[0].imshow(img, cmap="gray")
    # ax[1].imshow(img2, cmap="gray")
    # plt.show()