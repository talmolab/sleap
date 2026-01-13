# Evaluation Metrics

Here, we explain the various evaluation metrics we output at the end of running inference with a trained SLEAP model. We report 5 broad categories of metrics:

- [Distance metrics](#distance-metrics)
- [Object Keypoint similarity (OKS)](#object-keypoint-similarity-oks)
- [Percentage of Correct Keypoints (PCK) metrics](#percentage-of-correct-keypoints-pck-metrics)
- [Visibility metrics](#visibility-metrics)
- [VOC metrics](#voc-metrics)

## Distance Metrics

This metric computes the Euclidean distance between pairs of predicted and ground-truth (gt) instances. For each instance pair, we calculate the L2 norm of the difference between the predicted and corresponding ground-truth keypoints. The following statistics are reported:

- *`Avg_dist`*: Mean Euclidean distance across all (pred, gt) pairs.

- *`Dist@k`*: Percentile-based distance metrics which includes the distance at the 50th, 75th, 90th, 95th, and 99th percentiles (denoted as p50, p75, p90, p95, p99).

These metrics provide insight into the distribution of how far off the predictions are from the ground-truth key-points.


## Object-keypoint similarity (OKS)

This returns the mean OKS score between every pair of ground truth and
predicted instance, ranging from 0 to 1.0 and 1.0 indicating a perfect match. OKS provides a measure of similarity between ground-truth and precicted pose by taking into account the instance size (scale) and node visibility.

OKS is computed by measuring the Euclidean distance between each predicted keypoint and its corresponding ground-truth keypoint. This distance is then normalized based on the scale of the object (bounding box area for the instance) and standard-deviation that defines the spread in the localization accuracy of each node. For each node, keypoint similarity is computed by taking the negative expoenent of the normalized distance. Mean OKS is the average of keypoint similarities across all visible nodes.

The implementation is based off of the descriptions in: [Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation." ICCV (2017)](https://arxiv.org/abs/1707.05388).

## Percentage of Correct Keypoints (PCK) metrics

This metric measures the fraction of keypoints that fall within a certain pixel distance (threshold) from the ground-truth location. This is useful to evaluate how precise the predicted points are. The following are generated using PCK metric:

- *`PCKs`*: PCK on each predicted instances for each node at different thresholds. (Thresholds: [1, 2, 3, ..., 10])
- *`mPCK part`*: Mean PCK per node averaged over all predicted instances across thresholds.
- *`mPCK`*: Mean PCK across all predicted nodes and thresholds.

## Visibility metrics

This metric evaluates the visibility accuracy of the predicted nodes. It measures how well the model identifies whether the keypoint is present or missing - independent of its spatial accuracy (i.e. distance from ground-truth). This is useful for evaluating models on datasets with occlusions (NaN nodes).

The following statistics are computed across all matched instance pairs:

- *`True positives (TP)`*: Node is visible in both ground-truth and prediction.
- *`False positives (FP)`*: Node is missing in ground-truth but visible in prediction.
- *`True negatives (TN)`*: Node is missing in both ground-truth and prediction.
- *`False negatives (FN)`*: Node is visible in ground-truth but missing in prediction.
- *`Precision (TP / (TP + FP))`*: Proportion of predicted visible nodes that are correct.
- *`Recall (TP / (TP + FN))`*: Proportion of actual visible nodes that were correctly predicted.

## VOC metrics

The following VOC-style metrics are generated using either OKS or PCK as the matching scores and a set of thresholds where a predicted instance is considered as a True Postive if it's match score is greater than the threshold else it is counted as a False Positive. 

- *`Average Precision (AP)`*: Average of best precisions over fixed set of recall thresholds, at each match score threshold.
- *`Average Recall (AR)`*: Maximum recall achieved at each match score threshold.
- *`Mean Average Precision (mAP)`*: Mean of average precisions across match thresholds.
- *`Mean Average Recall (mAR)`*: Mean of average recalls across match thresholds.

To know more about how to generate these metrics using SLEAP APIs, take a look at the [model evaluation notebook](../notebooks/Model_evaluation.ipynb). 
Start evaluating your SLEAPs!