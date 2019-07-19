from sleap.io.video import Video
from sleap.io.dataset import Labels

import cv2
import numpy as np
import math
from time import time, clock

def save_labeled_video(filename, labels, video, frames, fps=15):
    output_size = (video.height, video.width)

    # Create frame images

    t0 = clock()
    total_count = len(frames)

    print(f"Writing video with {total_count} frame images...")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename, fourcc, fps, (video.width, video.height))

    chunk_size = 256
    chunk_count = math.ceil(total_count/chunk_size)

    for chunk_i in range(chunk_count):

        print(f"Chunk {chunk_i}/{chunk_count}")

        # Read the next chunk of frames
        frame_start = chunk_size * chunk_i
        frame_end = min(frame_start + chunk_size, total_count)
        frames_idx_chunk = frames[frame_start:frame_end]

        # Load frames from video
        section_t0 = clock()
        video_frame_images = video[frames_idx_chunk]
        print(f"    read time: {clock() - section_t0}")

        # Add overlays to each frame
        section_t0 = clock()
        imgs = []
        for i, frame_idx in enumerate(frames_idx_chunk):
            img = get_frame_image(video_frame=video_frame_images[i], frame_idx=frame_idx,
                                  height=video.height, width=video.width)

            imgs.append(img)
        print(f"    overlay time: {clock() - section_t0}")

        # Write frames to new video
        section_t0 = clock()
        for img in imgs:
            out.write(img)
        print(f"    write time: {clock() - section_t0}")

        elapsed = clock() - t0
        fps = frame_end/elapsed
        remaining_time = (total_count - frame_end)/fps
        print(f"Completed {frame_end}/{total_count} [{round(elapsed, 2)} s | {round(fps, 2)} FPS | approx {round(remaining_time, 2)} s remaining]")


    out.release()

    print(f"Done writing video [{clock() - t0} s]")

def img_to_cv(img):
    # Convert RGB to BGR for OpenCV
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert grayscale to BGR
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def get_frame_image(video_frame, frame_idx, width, height, overlay_callback=None):
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
