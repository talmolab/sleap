"""
Support for loading video frames (by chunk) in background process.
"""

from sleap import Video
from sleap.message import PairedSender, PairedReceiver

import cattr
import logging
import time
import numpy as np
from math import ceil
from multiprocessing import Process
from typing import Iterable, Iterator, List, Optional, Tuple


logger = logging.getLogger(__name__)


class AsyncVideo:
    """Supports fetching chunks from video in background process."""

    def __init__(self, base_port: int = 9010):
        self.base_port = base_port

        # Spawn the server as a background process
        self.server = AsyncVideoServer(self.base_port)
        self.server.start()

        # Create sender/receiver for sending requests and receiving data via ZMQ
        sender = PairedSender.from_tcp_ports(self.base_port, self.base_port + 1)
        result_receiver = PairedReceiver.from_tcp_ports(
            send_port=self.base_port + 2, rec_port=self.base_port + 3
        )

        sender.setup()
        result_receiver.setup()

        self.sender = sender
        self.receiver = result_receiver

        # Use "handshake" to ensure that initial messages aren't dropped
        self.handshake_success = sender.send_handshake()

    def close(self):
        """Close the async video server and communication ports."""
        if self.sender and self.server:
            self.sender.send_dict(dict(stop=True))
            self.server.join()

        self.server = None

        if self.sender:
            self.sender.close()
            self.sender = None

        if self.receiver:
            self.receiver.close()
            self.receiver = None

    def __del__(self):
        self.close()

    @classmethod
    def from_video(
        cls,
        video: Video,
        frame_idxs: Optional[Iterable[int]] = None,
        frames_per_chunk: int = 64,
    ) -> "AsyncVideo":
        """Create object and start loading frames in background process."""
        obj = cls()
        obj.load_by_chunk(
            video=video, frame_idxs=frame_idxs, frames_per_chunk=frames_per_chunk
        )
        return obj

    def load_by_chunk(
        self,
        video: Video,
        frame_idxs: Optional[Iterable[int]] = None,
        frames_per_chunk: int = 64,
    ):
        """
        Sends request for loading video in background process.

        Args:
            video: The :py:class:`Video` to load
            frame_idxs: Frame indices we want to load; if None, then full video
                is loaded.
            frames_per_chunk: How many frames to load per chunk.

        Returns:
            None, data should be accessed via :py:method:`chunks`.
        """
        # prime the video since this seems to make frames load faster (!?)
        video.test_frame

        request_dict = dict(
            video=cattr.unstructure(video), frames_per_chunk=frames_per_chunk
        )
        # if no frames are specified, whole video will be loaded
        if frame_idxs is not None:
            request_dict["frame_idxs"] = list(frame_idxs)

        # send the request
        self.sender.send_dict(request_dict)

    @property
    def chunks(self) -> Iterator[Tuple[List[int], np.ndarray]]:
        """
        Generator for fetching chunks of frames.

        When all chunks are loaded, closes the server and communication ports.

        Yields:
             Tuple with (list of frame indices, ndarray of frames)
        """
        done = False
        while not done:
            results = self.receiver.check_messages()
            if results:
                for result in results:
                    yield result["frame_idxs"], result["ndarray"]

                    if result["chunk"] == result["last_chunk"]:
                        done = True

        # automatically close when all chunks have been received
        self.close()


class AsyncVideoServer(Process):
    """
    Class which loads video frames in background on request.

    All interactions with video server should go through :py:class:`AsyncVideo`
    which runs in local thread.
    """

    def __init__(self, base_port: int):
        super(AsyncVideoServer, self).__init__()

        self.video = None
        self.base_port = base_port

    def run(self):
        receiver = PairedReceiver.from_tcp_ports(self.base_port + 1, self.base_port)
        receiver.setup()

        result_sender = PairedSender.from_tcp_ports(
            send_port=self.base_port + 3, rec_port=self.base_port + 2
        )
        result_sender.setup()

        running = True
        while running:
            requests = receiver.check_messages()
            if requests:

                for request in requests:

                    if "stop" in request:
                        running = False
                        logger.debug("stopping async video server")
                        break

                    if "video" in request:
                        self.video = cattr.structure(request["video"], Video)
                        logger.debug(f"loaded video: {self.video.filename}")

                    if self.video is not None:
                        if "frames_per_chunk" in request:

                            load_time = 0
                            send_time = 0

                            per_chunk = request["frames_per_chunk"]

                            frame_idxs = request.get(
                                "frame_idxs", list(range(self.video.frames))
                            )

                            frame_count = len(frame_idxs)
                            chunks = ceil(frame_count / per_chunk)

                            for chunk_idx in range(chunks):
                                start = per_chunk * chunk_idx
                                end = min(per_chunk * (chunk_idx + 1), frame_count)
                                chunk_frame_idxs = frame_idxs[start:end]

                                # load the frames
                                t0 = time.time()
                                frames = self.video[chunk_frame_idxs]
                                t1 = time.time()
                                load_time += t1 - t0

                                metadata = dict(
                                    chunk=chunk_idx,
                                    last_chunk=chunks - 1,
                                    frame_idxs=chunk_frame_idxs,
                                )

                                # send back results
                                t0 = time.time()
                                result_sender.send_array(metadata, frames)
                                t1 = time.time()
                                send_time += t1 - t0

                                logger.debug(f"returned chunk: {chunk_idx+1}/{chunks}")

                            logger.debug(f"total load time: {load_time}")
                            logger.debug(f"total send time: {send_time}")
                    else:
                        logger.warning(
                            "unable to process message since no video loaded"
                        )
                        logger.warning(request)
