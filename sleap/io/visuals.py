import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide2 import QtWidgets, QtGui

from sleap.io.video import Video
from sleap.io.dataset import Labels
from sleap.gui.video import GraphicsView, QtVideoPlayer, plot_instances, video_demo
from sleap.gui.confmapsplot import ConfMapsPlot, demo_confmaps
from sleap.gui.quiverplot import MultiQuiverPlot

import cv2
import numpy as np
from time import time, clock
import qimage2ndarray

def save_labeled_video(filename, labels, video, frames, fps=15, overlay_callback=None):
    output_size = (video.height, video.width)

    overlay_callback = overlay_callback or \
            (lambda scene, frame_idx:plot_instances(scene, frame_idx, labels))

    # Create frame images

    t0 = clock()
    total_count = len(frames)

    print(f"Writing video with {total_count} frame images...")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, fps, (video.width, video.height))

    for i, frame_idx in enumerate(frames):
        img = get_frame_image(video=video, frame_idx=frame_idx,
                              height=video.height, width=video.width,
                              overlay_callback=overlay_callback)
        # Convert RGB to BGR for OpenCV
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Convert grayscale to BGR
        elif img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if i > 0 and i%100 == 0:
            elapsed = clock() - t0
            fps = i/elapsed
            remaining_time = (total_count - i)/fps
            print(f"  frame {i}/{total_count} [{round(elapsed, 2)} s | {round(fps, 2)} FPS | approx {round(remaining_time, 2)} s remaining]")

        out.write(img)

    out.release()

    print(f"Done writing video [{clock() - t0} s]")

def get_frame_image(video, frame_idx, width, height, overlay_callback=None):
#     img = video.get_frame(frame_idx).copy()
    view = GraphicsView()
    view.scene.setSceneRect(0, 0, width, height)

    # video image
    video_frame_image = qimage2ndarray.array2qimage(video.get_frame(frame_idx))
    view.setImage(video_frame_image)

    if callable(overlay_callback):
        overlay_callback(view.scene, frame_idx)

    # convert graphicsview to ndarray
    qt_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
    qt_painter = QtGui.QPainter(qt_image)
    view.scene.render(qt_painter)
    qt_painter.end()
    img = qimage2ndarray.rgb_view(qt_image).copy()

    return img

def add_confmaps(scene, idx, confmaps):
    overlay_item = ConfMapsPlot(confmaps[idx,...])
    scene.addItem(overlay_item)

def add_pafs(scene, idx, pafs, decimation=10):
    overlay_item = MultiQuiverPlot(pafs[idx,...], decimation=decimation)
    scene.addItem(overlay_item)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='The output filename for the video')
    parser.add_argument('-f', '--fps', type=int, default=15,
                        help='Frames per second')
    args = parser.parse_args()

    labels = Labels.load_json(args.data_path)
    vid = labels.videos[0]

    app = QtWidgets.QApplication([])

    frames = [lf.frame_idx for lf in labels if len(lf.instances)]

#     if len(frames) > 2000:
#         frames = frames[:2000]

    print(f"Generating video with {len(frames)} labeled frames.")


    filename = args.output or args.data_path + ".avi"

    save_labeled_video(filename=filename,
                       labels=labels,
                       video=labels.videos[0],
                       frames=frames,
                       fps=args.fps)

    print(f"Video saved as: {filename}")
