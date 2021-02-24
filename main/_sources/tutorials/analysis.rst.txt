.. _export-analysis:

Export Data For Analysis
--------------------------

The easiest way to work with the data from SLEAP is to export an HDF5 file by choosing "**Export Analysis HDF5...**" in the "File" menu.

See :py:mod:`sleap.io.convert` for an explanation of the datasets inside the file.

MATLAB
~~~~~~

You can read this file in MATLAB like this::

    occupancy_matrix = h5read('path/to/analysis.h5','/track_occupancy')
    tracks_matrix = h5read('path/to/analysis.h5','/tracks')

See `here <https://www.mathworks.com/help/matlab/ref/h5read.html>`_ for more information about working with HDF5 files in MATLAB.

Python
~~~~~~

To read the file in Python you'll first need to install the `h5py package <http://docs.h5py.org/en/stable/>`_. You can then read data from the file like this::

    import h5py
    with h5py.File('path/to/analysis.h5', 'r') as f:
        occupancy_matrix = f['track_occupancy'][:]
        tracks_matrix = f['tracks'][:]

    print(occupancy_matrix.shape)
    print(tracks_matrix.shape)


**Note**: The datasets are stored column-major as expected by MATLAB. This means that if you're working with the file in Python you may want to first transpose the datasets so they match the shapes described in :py:mod:`sleap.io.convert`.
