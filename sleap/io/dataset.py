"""A LEAP Dataset represents annotated (labeled) video data.

A LEAP Dataset stores almost all data required for training of a model.
This includes, raw video frame data, labelled instances of skeleton _points,
confidence maps, part affinity fields, and skeleton data. A LEAP :class:`.Dataset`
is a high level API to these data structures that abstracts away their underlying
storage format.

"""

import logging
import h5py as h5
import os
import numpy as np
import attr
import cattr
import json

from typing import List, Dict, Union

from sleap.skeleton import Skeleton
from sleap.instance import Instance
from sleap.io.video import Video


@attr.s(auto_attribs=True)
class LabeledFrame:
    video: Video = attr.ib()
    frame_idx: int = attr.ib(converter=int)
    instances: List[Instance] = attr.ib(default=attr.Factory(list))

    @staticmethod
    def make_cattr():

        # We will need to serialize video references, so do the default.
        _cattr: cattr.Converter = Video.make_cattr()

        return _cattr


@attr.s(auto_attribs=True)
class Labels:
    """
    The LEAP :class:`.Labels` class represents an API for accessing labeled video
    frames and other associated metadata. This class is front-end for all
    interactions with loading, writing, and modifying these labels. The actual
    storage backend for the data is mostly abstracted away from the main
    interface.

    Args:
        instances: A list of instances
    """

    labels: List[LabeledFrame] = attr.ib(default=attr.Factory(list))

    def to_json(self):
        """
        Serialize all labels in the underling list of LabeledFrame(s) to a
        JSON structured string.

        Returns:
            The JSON representaiton of the string.
        """

        # Get the unique skeletons. Convert it to a list
        skeletons = set()
        for label in self.labels:
            for instance in label.instances:
                skeletons.add(instance.skeleton)
        skeletons = list(skeletons)

        # Get the unique videos. Convert it to a list
        videos = list({label.video for label in self.labels})

        # Register some unstructure hooks, start with video default cattr
        # since videos objects ar
        label_cattr = LabeledFrame.make_cattr()

        # By default label's cattr will serialize the skeleton and videos, override.
        # Don't serialize skeletons and videos within each video, store a
        # reference with a simple index to the two lists create above.
        label_cattr.register_unstructure_hook(Skeleton, lambda x: skeletons.index(x))
        label_cattr.register_unstructure_hook(Video, lambda x: videos.index(x))

        v = Video.make_cattr().unstructure(videos)

        # Serialize the skeletons, videos, and labels
        dicts = {
            'skeletons': Skeleton.make_cattr().unstructure(skeletons),
            'videos': Video.make_cattr().unstructure(videos),
            'labels': label_cattr.unstructure(self.labels)
         }

        return json.dumps(dicts)

    @staticmethod
    def save_json(labels: 'Labels', filename: str):
        json_str = labels.to_json()

        with open(filename, 'w') as file:
            file.write(json_str)

    @classmethod
    def from_json(cls, json_str: str):
        return cls()

    @classmethod
    def load_json(cls, filename: str):
        with open(filename, 'r') as file:
            return Labels.from_json(file.read())



