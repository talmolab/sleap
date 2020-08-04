"""
Group inference results ("examples") by frame.
"""

from collections import defaultdict


def group_examples(examples):
    """
    Group examples into dictionary.

    Key is (video_ind, frame_ind), val is list of examples matching key.
    """
    grouped_examples = defaultdict(list)
    for example in examples:
        video_ind = int(example["video_ind"].numpy().squeeze())
        frame_ind = int(example["frame_ind"].numpy().squeeze())
        grouped_examples[(video_ind, frame_ind)].append(example)
    return grouped_examples


def group_examples_iter(examples):
    """Iterator which groups examples.

    Yields ((video_ind, frame_ind), list of examples matching vid/frame).
    """
    last_key = None
    batch = []
    for example in examples:
        video_ind = int(example["video_ind"].numpy().squeeze())
        frame_ind = int(example["frame_ind"].numpy().squeeze())
        key = (video_ind, frame_ind)

        if last_key != key:
            if batch:
                yield last_key, batch
            last_key = key
            batch = [example]
        else:
            batch.append(example)

    if batch:
        yield key, batch


# @attr.s(auto_attribs=True)
# class ExampleGrouper:
#     grouping_key_list: List[Text] = ["video_ind", "frame_ind"]
#
#     @property
#     def input_keys(self) -> List[Text]:
#         """Return the keys that incoming elements are expected to have."""
#         return self.grouping_key_list
#
#     @property
#     def output_keys(self) -> List[Text]:
#         """Return the keys that outgoing elements will have."""
#         return self.grouping_key_list + ["grouped_examples"]
#
#     def transform_dataset(self, ds_input: tf.data.Dataset) -> tf.data.Dataset:
#         # The group_by_reducer op groups on a single tf.int64 key
#         # so we'll encode video_ind (upper 32 bits) and frame_ind (lower 32)
#
#         def encode_group_key(example):
#             # upper_bits = tf.bitwise.right_shift(
#             #     tf.dtypes.cast(example["video_ind"], tf.int64),
#             #     tf.constant([32], dtype=tf.int64)
#             # )
#             # lower_bits = tf.dtypes.cast(example["frame_ind"], tf.int64)
#             # example["group_key"] = upper_bits + lower_bits
#             example["group_key"] = tf.cast(example["video_ind"], tf.int64)
#             return example
#
#         # Here we apply the encoder
#         encoded_key_ds = ds_input.map(
#             encode_group_key, num_parallel_calls=tf.data.experimental.AUTOTUNE
#         )
#
#         # We'll want to "reduce" matching keys by creating a list of examples
#         reducer = tf.data.experimental.Reducer(
#             init_func=lambda _: np.int64(0.0),
#             reduce_func=lambda x, y: x + y,
#             finalize_func=lambda x: x)
#
#         # Now apply the grouping reduction
#         encoded_key_ds.apply(tf.data.experimental.group_by_reducer(
#             key_func=lambda example: example["group_key"],
#             reducer=reducer,
#         ))
#
#         return encoded_key_ds
