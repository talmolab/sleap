import attr
import numpy as np
import tensorflow as tf
from sleap.nn import peak_finding
from sleap.nn import model


class PeakFindingTests(tf.test.TestCase):
    def test_find_local_peaks(self):

        img_shape = [2, 8, 16, 3]
        peak_subs_gt = tf.constant(
            [
                [0, 2, 2, 0],  # (stop black from reformatting matrix)
                [0, 4, 6, 0],
                [1, 2, 2, 0],
                [1, 3, 2, 1],
            ],
        )
        peak_vals_gt = tf.ones(peak_subs_gt.shape[0])

        img = tf.scatter_nd(peak_subs_gt, peak_vals_gt, img_shape)

        peak_subs, peak_vals = peak_finding.find_local_peaks(img)

        self.assertAllEqual(peak_subs, peak_subs_gt)
        self.assertAllEqual(peak_vals, peak_vals_gt)

    def test_close_local_peaks(self):
        img_shape = [2, 8, 8, 3]

        peak_subs = tf.constant(
            [
                [0, 1, 1, 0],
                [0, 1, 2, 0],  # plateau (not considered local peak)
                [0, 7, 7, 1],
                [1, 1, 2, 1],
                [1, 2, 3, 2],
            ],
        )
        peak_vals = tf.ones(peak_subs.shape[0])

        img = tf.scatter_nd(peak_subs, peak_vals, img_shape)
        peaks, _ = peak_finding.find_local_peaks(img)

        self.assertLen(peaks, 3)

    def test_no_local_peaks(self):

        img_shape = [2, 8, 8, 3]
        img = tf.zeros(img_shape)

        peak_subs, peak_vals = peak_finding.find_local_peaks(img)

        self.assertEmpty(peak_subs)
        self.assertEmpty(peak_vals)

    def test_no_global_peaks(self):

        img_shape = [2, 8, 8, 3]
        img = tf.zeros(img_shape)

        peak_subs, peak_vals = peak_finding.find_global_peaks(img)

        img_channel_count = img_shape[0] * img_shape[3]

        # We expect the shape to be constant, even if there are no peaks
        self.assertLen(peak_subs, img_channel_count)
        self.assertLen(peak_vals, img_channel_count)

        # Peak locations should all be [0, 0]
        zeros = tf.zeros((img_channel_count, 2))
        self.assertAllEqual(zeros, peak_subs[:, 1:3])

    def test_local_peak_above_min(self):

        img_shape = [2, 8, 8, 3]

        peak_subs_raw = tf.constant(
            [
                [0, 2, 2, 0],  # (stop black from reformatting matrix)
                [0, 4, 6, 0],
                [1, 2, 2, 0],
                [1, 3, 2, 1],
            ],
        )
        peak_vals_raw = tf.constant([1.0, 2.0, 3.0, 4.0])

        img = tf.scatter_nd(peak_subs_raw, peak_vals_raw, img_shape)

        peak_subs, peak_vals = peak_finding.find_local_peaks(img, min_val=3.0)

        # Only the last peak should be above the min_val
        peak_subs_gt = peak_subs_raw[3:]
        peak_vals_gt = peak_vals_raw[3:]

        self.assertAllEqual(peak_subs, peak_subs_gt)
        self.assertAllEqual(peak_vals, peak_vals_gt)

    def test_find_global_peaks(self):

        img_shape = [2, 8, 16, 3]

        peak_subs_gt = tf.constant(
            [
                [0, 1, 2, 0],  # (stop black from reformatting matrix)
                [0, 2, 3, 1],
                [0, 3, 4, 2],
                [1, 5, 6, 0],
                [1, 7, 0, 1],
                [1, 1, 2, 2],
            ],
        )
        peak_vals_gt = tf.ones(peak_subs_gt.shape[0])

        img = tf.scatter_nd(peak_subs_gt, peak_vals_gt, img_shape)

        peak_subs, peak_vals = peak_finding.find_global_peaks(img)

        self.assertAllEqual(peak_subs, peak_subs_gt)
        self.assertAllEqual(peak_vals, peak_vals_gt)

    def test_find_global_peaks_from_multiple_local_peaks(self):

        img_shape = [2, 8, 16, 3]

        peak_subs_raw = tf.constant(
            [
                [0, 1, 2, 0],
                [0, 2, 3, 0],
                [0, 3, 4, 0],  # peak for image 0 channel 0
                # no peaks for image 0 channels 1 and 2
                [1, 5, 6, 0],  # peak for 1, 0
                [1, 7, 0, 1],  # peak for 1, 1
                [1, 1, 2, 2],  # peak for 1, 2
            ],
        )
        peak_vals_raw = tf.constant([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

        img = tf.scatter_nd(peak_subs_raw, peak_vals_raw, img_shape)

        peak_subs, peak_vals = peak_finding.find_global_peaks(img)

        peak_subs_gt = tf.constant(
            [
                [0, 3, 4, 0],  # peak for image 0 channel 0
                [0, 0, 0, 1],  # no peak for 0, 1 so expect 0, 0
                [0, 0, 0, 2],  # no peak for 0, 2 so expect 0, 0
                [1, 5, 6, 0],  # peak for 1, 0
                [1, 7, 0, 1],  # peak for 1, 1
                [1, 1, 2, 2],  # peak for 1, 2
            ],
        )
        peak_vals_gt = tf.constant([2.0, 0.0, 0.0, 2.0, 3.0, 3.0])

        self.assertAllEqual(peak_subs, peak_subs_gt)
        self.assertAllEqual(peak_vals, peak_vals_gt)

    def test_crop_centered_boxes(self):
        img_shape = [2, 4, 8, 3]

        img_len = np.product(img_shape)

        img = tf.reshape(tf.range(img_len), img_shape)
        print(img.shape)

        peaks = tf.constant(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 0, 2],  # top left corner
                [1, 3, 7, 0],  # bottom right corner
                # no peak in this channel
                [1, 3, 7, 2],
            ]
        )

        win_size = 3

        crops = peak_finding.crop_centered_boxes(img, peaks, win_size)

        crop_shape_gt = (peaks.shape[0], win_size, win_size, 1)

        self.assertAllEqual(crops.shape, crop_shape_gt)

        self.assertAllEqual(crops[0, ..., 0], img[0, 0:3, 0:3, 0])
        self.assertAllEqual(crops[1, ..., 0], img[0, 0:3, 0:3, 1])

        # Test for padded corner crop at min corner
        self.assertAllEqual(crops[2, 1:, 1:, 0], img[0, 0:2, 0:2, 2])

        # Test for padded corner crop at max corner
        self.assertAllEqual(crops[-1, :-1, :-1, 0], img[1, -2:, -2:, 2])

        # Make sure everything works if crop window is even and larger than img
        large_crops = peak_finding.crop_centered_boxes(img, peaks, 6)

        # Make sure that crop window was rounded up to nearest odd
        self.assertAllEqual(large_crops.shape, (peaks.shape[0], 7, 7, 1))

        # Make sure the large crops were correctly padded
        self.assertAllEqual(large_crops[0, -5:-1, -5:, 0], img[0, 0:5, 0:5, 0])
        self.assertAllEqual(large_crops[0, -1, :, 0], np.zeros((7,)))

    def test_make_gaussian_kernel(self):
        gaus = peak_finding.make_gaussian_kernel(9, 3)

        self.assertAllEqual(gaus.shape, (9, 9))

        # Make sure gaussian peak is at center
        self.assertAllEqual(tf.argmax(gaus), np.full((9,), 9 // 2))

    def test_smooth_imgs(self):
        img_shape = [2, 8, 8, 3]
        peak_subs = tf.constant(
            [
                [0, 1, 2, 0],  # (stop black from reformatting matrix)
                [0, 2, 3, 0],
                [0, 4, 5, 1],
                [1, 1, 2, 1],
                [1, 2, 3, 2],
            ],
        )
        peak_vals = tf.ones(peak_subs.shape[0])

        img = tf.scatter_nd(peak_subs, peak_vals, img_shape)

        smoothed = peak_finding.smooth_imgs(img, 3, 1)

        # Make sure two peaks were smoothed
        self.assertAllEqual(
            tf.argmax(smoothed[0, ..., 0]), np.array([0, 1, 1, 2, 2, 0, 0, 0])
        )

        # Make sure empty channel is still empty
        self.assertAllEqual(tf.argmax(smoothed[0, ..., 2]), np.zeros((8,)))

        # Make sure single peak was smoothed
        self.assertAllEqual(
            tf.argmax(smoothed[1, ..., 1]), np.array([0, 1, 1, 1, 0, 0, 0, 0])
        )

    def test_find_offsets_local_direction(self):
        # Test patch with asymmetry along x (column)
        patch = np.array(
            [
                [0.0, 1.0, 0.0],  # (stop black from reformatting matrix)
                [1.0, 3.0, 2.0],
                [0.0, 1.0, 0.0],
            ]
        ).reshape(1, 3, 3, 1)
        dirs = peak_finding.find_offsets_local_direction(patch, 0.25)
        self.assertAllEqual(dirs, np.array([[0.0, 0.25]]))

        # Test patch with asymmetry along y (row)
        patch = np.array(
            [
                [0.0, 2.0, 0.0],  # (stop black from reformatting)
                [1.0, 3.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ).reshape(1, 3, 3, 1)
        dirs = peak_finding.find_offsets_local_direction(patch, 0.25)
        self.assertAllEqual(dirs, np.array([[-0.25, 0.0]]))

        # Test patch with x and y asymmetry
        # Note that size of gradient doesn't affect result, just direction
        patch = np.array(
            [
                [0.0, 0.0, 0.0],  # (stop black from reformatting matrix)
                [0.0, 2.0, 1.0],
                [0.0, 0.5, 0.0],
            ]
        ).reshape(1, 3, 3, 1)
        dirs = peak_finding.find_offsets_local_direction(patch, 1.0)
        self.assertAllEqual(dirs, np.array([[1.0, 1.0]]))

        # Test symmetric patch
        patch = np.array(
            [
                [0.0, 1.0, 0.0],  # (stop black from reformatting matrix)
                [1.0, 3.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ).reshape(1, 3, 3, 1)
        dirs = peak_finding.find_offsets_local_direction(patch, 0.25)
        self.assertAllEqual(dirs, np.array([[0.0, 0.0]]))


class PeakRefinementTests(tf.test.TestCase):
    def setUp(self):
        img_shape = [2, 8, 8, 3]
        peak_subs = tf.constant(
            [
                [0, 1, 1, 0],
                [0, 1, 2, 0],  # make this higher so it's not a plateau
                [0, 2, 1, 0],
                [0, 2, 2, 0],
                [0, 7, 7, 1],
                [1, 1, 2, 1],
                [1, 2, 3, 2],
            ],
        )
        peak_vals = tf.constant([1, 1.1, 1, 1, 1, 1, 1])

        self.img = tf.scatter_nd(peak_subs, peak_vals, img_shape)
        self.peaks, _ = peak_finding.find_local_peaks(self.img)

    def test_refine_peaks_local_direction(self):
        refined = peak_finding.refine_peaks_local_direction(self.img, self.peaks)

        refined_gt = np.array(
            [
                [0.0, 1.25, 1.75, 0.0],
                [0.0, 7.0, 7.0, 1.0],
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 2.0],
            ]
        )

        self.assertAllEqual(refined, refined_gt)

    def test_delta_refine_local(self):
        refined = peak_finding.refine_peaks_local_direction(
            self.img, self.peaks, delta=0.5
        )

        refined_gt = np.array(
            [
                [0.0, 1.5, 1.5, 0.0],
                [0.0, 7.0, 7.0, 1.0],
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 2.0],
            ]
        )

        self.assertAllEqual(refined, refined_gt)


@attr.s(auto_attribs=True)
class PassthroughModel(model.InferenceModel):
    job: "sleap.nn.training.TrainingJob" = None

    def __attrs_post_init__(self):
        from sleap.nn import training
        from sleap.nn.architectures.leap import LeapCNN as backbone

        self._keras_model = lambda x: x

        training_model = training.Model(
            output_type=training.ModelOutputType.CONFIDENCE_MAP,
            backbone=backbone(),
            skeletons=[],
        )

        train_run = training.TrainingJob(
            model=training_model,
            trainer=training.Trainer(),
            save_dir="foo",
            run_name="foo",
        )

        self.job = train_run

    def predict(self, X, *args, **kwargs):
        return X


class PeakFindingIntegrationTests(tf.test.TestCase):
    def skip_test_peaking_finding_integration(self):
        img_shape = [2, 8, 8, 3]
        peak_subs = tf.constant(
            [
                [0, 1, 1, 0],  # (stop black from reformatting matrix)
                [0, 1, 2, 0],
                [0, 0, 0, 1],
                [1, 1, 2, 1],
                [1, 2, 3, 2],
            ],
        )
        peak_vals = tf.ones(peak_subs.shape[0])

        img = tf.scatter_nd(peak_subs, peak_vals, img_shape)

        x, y = peak_finding.ConfmapPeakFinder(PassthroughModel()).predict(img * 100)

        gt = np.array(
            [
                [0.0, 0.25, 0.25, 1.0],
                [0.0, 1.0, 1.25, 0.0],
                [0.0, 1.0, 1.75, 0.0],
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 2.0],
            ]
        )

        self.assertAllEqual(x, gt)

    def skip_test_peak_finding_zeros(self):
        img_shape = [2, 8, 8, 3]
        img = tf.zeros(img_shape)

        x, y = peak_finding.ConfmapPeakFinder(PassthroughModel()).predict(img)

        # Make sure there are no peaks when image is all zeros
        self.assertEmpty(x)
