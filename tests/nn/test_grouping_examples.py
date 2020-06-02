import tensorflow as tf
import numpy as np
from sleap.nn.data.grouping import group_examples, group_examples_iter


class DummyTfVal(object):
    def __init__(self, val):
        self._val = np.array(val)

    def numpy(self):
        return self._val


def make_examples():
    examples = []

    def add_example(video_ind, frame_ind, x):
        examples.append(
            {
                "video_ind": DummyTfVal(video_ind),
                "frame_ind": DummyTfVal(frame_ind),
                "x": x,
            }
        )

    add_example(video_ind=0, frame_ind=0, x=1)
    add_example(video_ind=0, frame_ind=0, x=2)
    add_example(video_ind=0, frame_ind=1, x=3)
    add_example(video_ind=1, frame_ind=0, x=4)
    add_example(video_ind=1, frame_ind=1, x=5)
    add_example(video_ind=1, frame_ind=1, x=6)

    return examples


def check_grouped_examples(grouped):
    assert len(grouped.keys()) == 4
    assert len(grouped[(0, 0)]) == 2
    assert len(grouped[(0, 1)]) == 1
    assert len(grouped[(1, 0)]) == 1
    assert len(grouped[(1, 1)]) == 2

    assert grouped[(1, 1)][0]["x"] == 5
    assert grouped[(1, 1)][1]["x"] == 6


def test_group_examples():
    examples = make_examples()
    grouped = group_examples(examples)
    check_grouped_examples(grouped)


def test_group_iterator():
    examples = make_examples()

    # Use iterator to build grouped dict
    grouped = dict()
    for key, val in group_examples_iter(examples):
        grouped[key] = val

    check_grouped_examples(grouped)


# def test_ds_grouping():
#     import numpy as np
#     ds = tf.data.Dataset.from_tensor_slices(
#         {
#             "video_ind": [0, 0, 0, 1, 1, 1],
#             "frame_ind": [0, 0, 1, 0, 1, 1],
#             # "x": [1, 2, 3, 4, 5, 6]
#         }
#     )
#
#     # reducer = tf.data.experimental.Reducer(
#     #     init_func=lambda _: np.int64(0),
#     #     reduce_func=lambda x, y: x + y,
#     #     finalize_func=lambda x: x)
#     # for i in range(1, 11):
#     #     dataset = tf.data.Dataset.range(2 * i).apply(
#     #         tf.data.experimental.group_by_reducer(lambda x: x % 2, reducer))
#
#     grouped_ds = ExampleGrouper().transform_dataset(ds)
#
#     print(list(grouped_ds.as_numpy_iterator()))
#     assert False
