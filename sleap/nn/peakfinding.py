import cv2
import numpy as np

def impeaksnms_cv(I, min_thresh=0.3, sigma=3, return_val=True):
    """ Find peaks via non-maximum suppresion using OpenCV. """

    # Threshold
    if min_thresh is not None:
        I[I < min_thresh] = 0

    # Blur
    if sigma is not None:
        I = cv2.GaussianBlur(I, (9,9), sigma)

    # Maximum filter
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]]).astype("uint8")
    m = cv2.dilate(I, kernel)

    # Convert to points
    r, c = np.nonzero(I > m)
    pts = np.stack((c, r), axis=1)

    # Return
    if return_val:
        vals = np.array([I[pt[1],pt[0]] for pt in pts])
        return pts.astype("float32"), vals
    else:
        return pts.astype("float32")


def find_all_peaks(confmaps, min_thresh=0.3, sigma=3):
    """ Finds peaks for all frames/channels in a stack of confidence maps """
    peaks = []
    peak_vals = []
    for confmap in confmaps:
        peaks_i = []
        peak_vals_i = []
        for i in range(confmap.shape[-1]):
            peak, val = impeaksnms_cv(confmap[...,i], min_thresh=min_thresh, sigma=sigma, return_val=True)
            peaks_i.append(peak)
            peak_vals_i.append(val)
        peaks.append(peaks_i)
        peak_vals.append(peak_vals_i)

    return peaks, peak_vals

def find_all_single_peaks(confmaps, min_thresh=0.3):
    """
    Finds single peak for each frame/channel in a stack of conf maps.

    Returns:
        list of points array for each confmap
        each points array is N(=channels) x 3 for x, y, peak val
    """
    all_point_arrays = []

    for confmap in confmaps:
        peaks_vals = [image_single_peak(confmap[...,i], min_thresh) for i in range(confmap.shape[-1])]
        peaks_vals = [(*point, val) for point, val in peaks_vals]
        points_array = np.stack(peaks_vals, axis=0)
        all_point_arrays.append(points_array)

    return all_point_arrays

def image_single_peak(I, min_thresh):
    peak = np.unravel_index(I.argmax(), I.shape)
    val = I[peak]

    if val < min_thresh:
        y, x = (np.nan, np.nan)
        val = np.nan
    else:
        y, x = peak

    return (x, y), val