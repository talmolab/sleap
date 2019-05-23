import numpy as np
import tensorflow as tf
import keras
import imgaug

class Augmenter(keras.utils.Sequence):
    def __init__(self, X, Y, output_names=None, batch_size=32, shuffle=True, rotation=(-180, 180), scale=None):
        self.X = X
        self.Y = Y
        self.output_names = output_names
        self.num_outputs = 1 if output_names is None else len(output_names)
        self.batch_size = batch_size
        self.num_samples = len(X)
        self.rotation = rotation
        self.scale = scale

        # Setup batching
        all_idx = np.arange(self.num_samples)
        self.batches = np.array_split(all_idx, np.ceil(self.num_samples / self.batch_size))
        
        # Initial shuffling
        if shuffle:
            self.shuffle()

        # Setup affine augmentation
        # TODO: translation?
        self.aug_stack = []
        if self.rotation is not None:
            self.rotation = rotation if isinstance(rotation, tuple) else (-rotation, rotation)
            if self.scale is not None and self.scale[0] != self.scale[1]:
                self.scale = (min(self.scale), max(self.scale))
                self.aug_stack.append(imgaug.augmenters.Affine(rotate=self.rotation, scale=self.scale))
            else:
                self.aug_stack.append(imgaug.augmenters.Affine(rotate=self.rotation))
        
        # TODO: Flips?
        # imgaug.augmenters.Fliplr(0.5)

        # Create augmenter
        self.aug = imgaug.augmenters.Sequential(self.aug_stack)
        
    def shuffle(self, batches_only=False):
        """ Index-based shuffling """

        if batches_only:
            # Shuffle batch order
            np.random.shuffle(self.batches)
        else:
            # Re-batch after shuffling
            all_idx = np.arange(self.num_samples)
            np.random.shuffle(all_idx)
            self.batches = np.array_split(all_idx, np.ceil(self.num_samples / self.batch_size))
        
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, batch_idx):
        idx = self.batches[batch_idx]
        X = self.X[idx].copy()
        Y = self.Y[idx].copy()
        
        aug_det = self.aug.to_deterministic()
        X = aug_det.augment_images(X)
        Y = aug_det.augment_images(Y)
        
        if self.output_names is not None:
            Y = {output_name: Y for output_name in self.output_names}
        
        return (X, Y)


if __name__ == "__main__":
    import os
    from sleap.io.labels import Labels
    from sleap.nn.datagen import generate_images, generate_confidence_maps

    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    if not os.path.exists(data_path):
        data_path = "D:/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    
    labels = Labels(data_path)
    imgs, keys = generate_images(labels)
    confmaps, _keys, points = generate_confidence_maps(labels)

    aug = Augmenter(X=imgs, Y=confmaps)

    import matplotlib.pyplot as plt

    x, y = aug[0]

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x[0].squeeze())

    plt.subplot(1,2,2)
    plt.imshow(y[0].max(axis=-1).squeeze())

    plt.show()
