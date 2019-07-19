from sleap.io.video import Video
from sleap.io.dataset import Labels

import cv2
import numpy as np
import math
from time import time, clock

from queue import Queue
from threading import Thread

# Object that signals shutdown
_sentinel = object()

def reader(out_q, video, frames):

    total_count = len(frames)
    chunk_size = 64
    chunk_count = math.ceil(total_count/chunk_size)

    print(f"Chunks: {chunk_count}, chunk size: {chunk_size}")

    i = 0
    for chunk_i in range(chunk_count):

        # Read the next chunk of frames
        frame_start = chunk_size * chunk_i
        frame_end = min(frame_start + chunk_size, total_count)
        frames_idx_chunk = frames[frame_start:frame_end]

        t0 = clock()

        # Load frames from video
        video_frame_images = video[frames_idx_chunk]

        elapsed = clock() - t0
        fps = len(frames_idx_chunk)/elapsed
        print(f"reading chunk {i} in {elapsed} s = {fps} fps")
        i += 1

        out_q.put((frames_idx_chunk, video_frame_images))

    out_q.put(_sentinel)

def marker(in_q, out_q, labels):
    chunk_i = 0
    while True:
        data = in_q.get()

        if data is _sentinel:
            in_q.put(_sentinel)
            break

        frames_idx_chunk, video_frame_images = data

        t0 = clock()
        imgs = []
        for i, frame_idx in enumerate(frames_idx_chunk):
            img = get_frame_image(
                        video_frame=video_frame_images[i],
                        frame_idx=frame_idx,
                        labels=labels)

            imgs.append(img)
        elapsed = clock() - t0
        fps = len(imgs)/elapsed
        print(f"marking chunk {chunk_i} in {elapsed} s = {fps} fps")
        chunk_i += 1
        out_q.put(imgs)

    out_q.put(_sentinel)

def writer(in_q, filename, fps, img_w_h):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, fps, img_w_h)

    i = 0
    while True:
        data = in_q.get()

        if data is _sentinel:
            in_q.put(_sentinel)
            break

        t0 = clock()
        for img in data:
            out.write(img)
        elapsed = clock() - t0
        fps = len(data)/elapsed
        print(f"writing chunk {i} in {elapsed} s = {fps} fps")
        i += 1

    out.release()

def save_labeled_video(filename, labels, video, frames, fps=15):
    output_size = (video.height, video.width)

    print(f"Writing video with {len(frames)} frame images...")

    t0 = clock()

    q1 = Queue()
    q2 = Queue()

    thread_read = Thread(target=reader, args=(q1, video, frames,))
    thread_mark = Thread(target=marker, args=(q1, q2, labels,))
    thread_write = Thread(target=writer, args=(q2, filename, fps, (video.width, video.height)))

    thread_read.start()
    thread_mark.start()
    thread_write.start()
    thread_write.join()

    elapsed = clock() - t0
    fps = len(frames)/elapsed
    print(f"Done in {elapsed} s, fps = {fps}.")

def img_to_cv(img):
    # Convert RGB to BGR for OpenCV
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert grayscale to BGR
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def get_frame_image(video_frame, frame_idx, labels):
    img = img_to_cv(video_frame)
    plot_instances_cv(img, frame_idx, labels)
    return img

def _point_int_tuple(point):
    return int(point.x), int(point.y)

def plot_instances_cv(img, frame_idx, labels):
    cmap = ([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ])
    lfs = [label for label in labels.labels if label.video == labels.videos[0] and label.frame_idx == frame_idx]

    if len(lfs) == 0: return

    count_no_track = 0
    for i, instance in enumerate(lfs[0].instances_to_show):
        if instance.track in labels.tracks:
            track_idx = labels.tracks.index(instance.track)
        else:
            # Instance without track
            track_idx = len(labels.tracks) + count_no_track
            count_no_track += 1

        inst_color = cmap[track_idx%len(cmap)]

        plot_instance_cv(img, instance, inst_color)

def plot_instance_cv(img, instance, color, marker_radius=4):

    # RGB -> BGR for cv2
    cv_color = color[::-1]

    for (node, point) in instance.nodes_points:
        # plot node at point
        if point.visible and not point.isnan():
            cv2.circle(img,
                    _point_int_tuple(point),
                    marker_radius,
                    cv_color,
                    lineType=cv2.LINE_AA)
    for (src, dst) in instance.skeleton.edges:
        # Make sure that both nodes are present in this instance before drawing edge
        if src in instance and dst in instance:
            if instance[src].visible and instance[dst].visible \
                and not instance[src].isnan() and not instance[dst].isnan():

                cv2.line(
                        img,
                        _point_int_tuple(instance[src]),
                        _point_int_tuple(instance[dst]),
                        cv_color,
                        lineType=cv2.LINE_AA)

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

    frames = [lf.frame_idx for lf in labels if len(lf.instances)]

#     if len(frames) > 20:
#         frames = frames[:20]

    filename = args.output or args.data_path + ".avi"

    save_labeled_video(filename=filename,
                       labels=labels,
                       video=labels.videos[0],
                       frames=frames,
                       fps=args.fps)

    print(f"Video saved as: {filename}")
