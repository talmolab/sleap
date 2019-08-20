import attr
import cv2
import numpy as np
from typing import List, Optional, Tuple

from sleap.nn.datagen import _bbs_from_points, _pad_bbs, _crop

@attr.s(auto_attribs=True, slots=True)
class DataTransform:
    """
    Object to apply transformations during inference while keeping track so that
    we can easily apply inverse transform to points returned by inference.
    """

    scale: float = 1.0
    frame_idxs: List[int] = attr.ib(default=attr.Factory(list))
    bounding_boxes: List = attr.ib(default=attr.Factory(list))
    crop_size: Optional[int] = None
    is_cropped: bool = False
    
    def _init_frame_idxs(self, frame_count):
        if len(self.frame_idxs) == 0:
            self.frame_idxs = list(range(frame_count))

    def get_data_idxs(self, frame_idx):
        return [i for i in range(len(self.frame_idxs)) if self.frame_idxs[i] == frame_idx]

    def get_frame_idxs(self, idxs):
        if type(idxs) == int:
            return self._safe_frame_idx(idxs)
        else:
            return [self._safe_frame_idx(idx) for idx in idxs]
            
    def _safe_frame_idx(self, idx: int):
        return self.frame_idxs[idx] if idx < len(self.frame_idxs) else idx
    
    def scale_to(self, imgs, target_size):
        """
        Scale images to a specified (h, w) target size.
        If target size matches current image size, we don't do anything.
        
        Args:
            imgs: ndarray with shape (count, height, width, channels)
            target_size: (h, w) tuple
        Returns:
            images scaled to target size
        """
        img_count, img_h, img_w, img_channels = imgs.shape
        h, w = target_size

        self._init_frame_idxs(img_count)

        # Resize the frames to the target_size if not current size
        if (img_h, img_w) != target_size:

            # build ndarray for new size
            scaled_imgs = np.zeros((imgs.shape[0], h, w, imgs.shape[3]))
            for i in range(imgs.shape[0]):
                # resize using cv2
                img = cv2.resize(imgs[i, :, :], (w, h))
                 # add back singleton channel (removed by cv2)
                if img_channels == 1:
                    img = img[..., None]
                else:
                    # Although cv2 uses BGR order for channels, we shouldn't have to
                    # swap order of color channels because a resize doesn't change it.
                    # If we did have to convert BGR -> RGB, then we'd do this:
                    # img = img[..., ::-1]
                    pass
                scaled_imgs[i, ...] = img
        else:
            scaled_imgs = imgs
        
        # update object state (so we can invert)
        self.scale = self.scale * (h/img_h)
        
        return scaled_imgs

    def centroid_crop(self, imgs: np.ndarray, centroids: list, crop_size: int=0):
        """
        Crop images around centroid points.
        Updates state of DataTransform object so we can later invert on points.

        Args:
            imgs: ndarray with shape (count, height, width, channels)
            centroids: list of frames -> instances -> centroids as (x,y) row vector
            crop_size: int, the crop square side length
        Returns:
            imgs, first dimension will be larger if more than one centroid per frame
        """
        img_count = imgs.shape[0]
        img_shape = imgs.shape[1], imgs.shape[2]

        self._init_frame_idxs(img_count)

        # List of bounding box for every instance, map from list idx -> frame idx
        bbs, idxs = _bbs_from_points(centroids)

        # Grow all bounding boxes to the same size
        bbs = _pad_bbs(bbs, (crop_size, crop_size), img_shape)

        # Crop images
        imgs = _crop(imgs, idxs, bbs)

        self.bounding_boxes = bbs
        self.frame_idxs = list(map(lambda i: self.frame_idxs[i], idxs))
        self.crop_size = crop_size
        self.is_cropped = True

        return imgs

    def invert(self, idx: int, point_array: np.ndarray) -> np.ndarray:
        """
        Map points in a transformed image back on to the original image.
        
        Args:
            idx: the index of the image (first dimension of transformed imgs)
            point_array: a 2d ndarray with (x, y) row for each point
        Returns:
            point_array with points on original image
        """
        # unscale
        new_point_array = point_array / self.scale

        if idx < len(self.bounding_boxes):
            # translate point_array using corresponding bounding_box
            bb = self.bounding_boxes[idx]
        
            top_left_point = ((bb[0], bb[1]),) # for (x, y) row vector
            new_point_array += np.array(top_left_point)
                
        return new_point_array