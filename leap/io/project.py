"""
A LEAP project collects video data, skeletons, models, training data, into a
single data structure. It is backed by an assortment of data files stored in
a single directory. It can represent several sessions of work using the LEAP
pipeline and GUI and is the main data structure for all inputs and outputs
to the algorithms and GUI.
"""


class Project:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

        # Load any datasets present in the project directory into memory

    def _load_dataset(data_path, X_dset="box", Y_dset="confmaps", permute=(0, 3, 2, 1)):
        """ Loads and normalizes datasets. """

        # Load
        t0 = time()
        with h5py.File(data_path, "r") as f:
            X = f[X_dset][:]
            Y = f[Y_dset][:]
        print("Loaded %d samples [%.1fs]" % (len(X), time() - t0))

        # Adjust dimensions
        t0 = time()
        X = preprocess(X, permute)
        Y = preprocess(Y, permute)
        print("Permuted and normalized data. [%.1fs]" % (time() - t0))

        return X, Y