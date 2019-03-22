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

        # Register some unstructure hooks since we don't want complete deserialization
        # of video and skeleton objects present in the labels. We will serialize these
        # as references to the above constructed lists to limit redundant data in the
        # json
        label_cattr = cattr.Converter()

        # By default label's cattr will serialize the skeleton and videos, override.
        # Don't serialize skeletons and videos within each video, store a
        # reference with a simple index to the two lists create above.
        label_cattr.register_unstructure_hook(Skeleton, lambda x: skeletons.index(x))
        label_cattr.register_unstructure_hook(Video, lambda x: videos.index(x))

        # Serialize the skeletons, videos, and labels
        dicts = {
            'skeletons': Skeleton.make_cattr().unstructure(skeletons),
            'videos': cattr.unstructure(videos),
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

        dicts = json.loads(json_str)

        # First, deserialize the skeleton and videos lists, the labels reference these
        # so we will need them while deserializing.
        skeletons = Skeleton.make_cattr().structure(dicts['skeletons'], List[Skeleton])
        videos = Skeleton.make_cattr().structure(dicts['videos'], List[Video])

        @attr.s(auto_attribs=True)
        class SkeletonRef:
            idx: int = attr.ib()

        @attr.s(auto_attribs=True)
        class VideoRef:
            idx: int = attr.ib()

        label_cattr = cattr.Converter()
        label_cattr.register_structure_hook(Skeleton, lambda x,type: skeletons[x])
        label_cattr.register_structure_hook(Video, lambda x,type: videos[x])
        labels = label_cattr.structure(dicts['labels'], List[LabeledFrame])

        return cls(labels=labels)

    @classmethod
    def load_json(cls, filename: str):
        with open(filename, 'r') as file:
            return Labels.from_json(file.read())



