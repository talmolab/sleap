"""Custom loss functions and metrics."""

import tensorflow as tf
from sleap.nn.config import HardKeypointMiningConfig


def compute_ohkm_loss(
    y_gt: tf.Tensor,
    y_pr: tf.Tensor,
    hard_to_easy_ratio: float = 2.0,
    min_hard_keypoints: int = 2,
    max_hard_keypoints: int = -1,
    loss_scale: float = 5.0,
) -> tf.Tensor:
    """Compute the online hard keypoint mining loss."""
    # Compute elementwise squared difference.
    loss = tf.math.squared_difference(y_gt, y_pr)  # rank 4

    # Store initial shape for normalization.
    batch_shape = tf.shape(loss)

    # Reduce over everything but channels axis.
    loss = tf.reduce_sum(loss, axis=[0, 1, 2])

    # Compute the loss for the "easy" keypoint.
    best_loss = tf.math.reduce_min(loss)

    # Find the number of hard keypoints.
    is_hard_keypoint = (loss / best_loss) >= hard_to_easy_ratio
    n_hard_keypoints = tf.reduce_sum(tf.cast(is_hard_keypoint, tf.int32))

    # Work out the actual final number of keypoints to consider as hard.
    if max_hard_keypoints < 0:
        max_hard_keypoints = tf.shape(loss)[0]
    else:
        max_hard_keypoints = tf.minimum(max_hard_keypoints, tf.shape(loss)[0])
    k = tf.minimum(tf.maximum(n_hard_keypoints, min_hard_keypoints), max_hard_keypoints)

    # Pull out the top hard values.
    k_vals, k_inds = tf.math.top_k(loss, k=k, sorted=False)

    # Apply weights.
    k_loss = k_vals * loss_scale

    # Reduce over all channels.
    n_elements = tf.cast(
        batch_shape[0] * batch_shape[1] * batch_shape[2] * k, tf.float32
    )
    k_loss = tf.reduce_sum(k_loss) / n_elements

    return k_loss


class OHKMLoss(tf.keras.losses.Loss):
    """Online hard keypoint mining loss.

    This loss serves to dynamically reweight the MSE of the top-K worst channels in each
    batch. This is useful when fine tuning a model to improve performance on a hard
    part to optimize for (e.g., small, hard to see, often not visible).

    Note: This works with any type of channel, so it can work for PAFs as well.

    Attributes:
        hard_to_easy_ratio: The minimum ratio of the individual keypoint loss with
            respect to the lowest keypoint loss in order to be considered as "hard".
            This helps to switch focus on across groups of keypoints during training.
        min_hard_keypoints: The minimum number of keypoints that will be considered as
            "hard", even if they are not below the `hard_to_easy_ratio`.
        max_hard_keypoints: The maximum number of hard keypoints to apply scaling to.
            This can help when there are few very easy keypoints which may skew the
            ratio and result in loss scaling being applied to most keypoints, which can
            reduce the impact of hard mining altogether.
        loss_scale: Factor to scale the hard keypoint losses by.
    """

    def __init__(
        self,
        hard_to_easy_ratio: float = 2.0,
        min_hard_keypoints: int = 2,
        max_hard_keypoints: int = -1,
        loss_scale: float = 5.0,
        name="ohkm",
        **kwargs
    ):
        super(OHKMLoss, self).__init__(name=name, **kwargs)
        self.hard_to_easy_ratio = hard_to_easy_ratio
        self.min_hard_keypoints = min_hard_keypoints
        self.max_hard_keypoints = max_hard_keypoints
        self.loss_scale = loss_scale

    @classmethod
    def from_config(cls, config: HardKeypointMiningConfig) -> "OHKMLoss":
        return cls(
            hard_to_easy_ratio=config.hard_to_easy_ratio,
            min_hard_keypoints=config.min_hard_keypoints,
            max_hard_keypoints=config.max_hard_keypoints
            if config.max_hard_keypoints is not None
            else -1,
            loss_scale=config.loss_scale,
        )

    def call(self, y_gt, y_pr, sample_weight=None):
        return compute_ohkm_loss(
            y_gt,
            y_pr,
            hard_to_easy_ratio=self.hard_to_easy_ratio,
            min_hard_keypoints=self.min_hard_keypoints,
            max_hard_keypoints=self.max_hard_keypoints,
            loss_scale=self.loss_scale,
        )


class PartLoss(tf.keras.metrics.Metric):
    """Compute channelwise loss.

    Useful for monitoring the MSE for specific body parts (channels).

    Attributes:
        channel_ind: Index of channel to compute MSE for.
        name: Name of the loss tensor.
    """

    def __init__(self, channel_ind, name="part_loss", **kwargs):
        super(PartLoss, self).__init__(name=name, **kwargs)
        self.channel_ind = channel_ind
        self.channel_mse = self.add_weight(
            name=name + ".mse", initializer="zeros", dtype=tf.float32
        )
        self.n_samples = self.add_weight(
            name=name + ".n_samples", initializer="zeros", dtype=tf.int32
        )
        self.height = self.add_weight(
            name=name + ".height", initializer="zeros", dtype=tf.int32
        )
        self.width = self.add_weight(
            name=name + ".width", initializer="zeros", dtype=tf.int32
        )

    def update_state(self, y_gt, y_pr, sample_weight=None):
        shape = tf.shape(y_gt)
        n_samples = shape[0]
        channel_mse = tf.reduce_sum(
            tf.math.squared_difference(
                tf.gather(y_gt, self.channel_ind, axis=3),
                tf.gather(y_pr, self.channel_ind, axis=3),
            )
        )  # rank 4

        self.height.assign(shape[1])
        self.width.assign(shape[2])
        self.n_samples.assign_add(n_samples)
        self.channel_mse.assign_add(channel_mse)

    def result(self):
        return self.channel_mse / tf.cast(
            self.n_samples * self.height * self.width, tf.float32
        )
