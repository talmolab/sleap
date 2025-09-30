# 8. Exporting the results

SLEAP stores labeled data, predictions, and all metadata in `.slp` files. These files use a custom format that is optimized for SLEAP workflows and contain metadata, but ultimately are just HDF5 files that implement our [data model](../notebooks/Data_structures.ipynb).

These `.slp` files are not intended for final use in analysis since they require SLEAP to be parsed appropriately.

Once you're ready to analyze your results, you have several options for different file formats for export. While these do not contain all the metadata used during the labeling and training stage, they are more convenient for analysis and portable for use in different downstream softwares and libraries.

The main options we provide are listed below. We recommend exporting your proofread predictions to each of these formats so you can explore your data in the next step of the tutorial.

## NWB

The <a href="https://www.nwb.org/" target="_blank">Neurodata Without Borders (NWB) </a> format provides a data standard for describing and storing neural and behavioral data.

We strongly recommend using NWB as it affords tremendous advantages in terms of portability, standardization, reproducibility and compatibility with a broader scientific software ecosystem. For example, NWB formatted data can be hosted for free in the <a href="https://www.dandiarchive.org/" target="_blank">DANDI Archive </a>.

SLEAP <a href="https://nwb-overview.readthedocs.io/en/latest/tools/sleap/sleap.html" target="_blank">natively supports NWB</a> through the <a href="https://github.com/rly/ndx-pose" target="_blank">ndx-pose extension</a>.

For more information on using NWB in Python or MATLAB, check out the guide on <a href="https://nwb-overview.readthedocs.io/en/latest/file_read/file_read.html" target="_blank">reading NWB files</a>.

To export to NWB in SLEAP, go to **File** → **Export NWB...**

## HDF5

The <a href="https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html" target="_blank">Hierarchical Data Format v5 (HDF5)</a> is a widely used data format for storing numerical data such as N-dimensional arrays and is widely employed in scientific computing.

All major programming languages support HDF5, including <a href="https://docs.h5py.org/en/latest/index.html" target="_blank">Python</a>, <a href="https://cran.r-project.org/web/packages/hdf5r/index.html" target="_blank">R</a>, and <a href="https://www.mathworks.com/help/matlab/hdf5-files.html" target="_blank">MATLAB</a>.

We recommend <a href="https://docs.h5py.org/en/latest/quick.html#core-concepts" target="_blank">this guide</a> for getting started with HDF5 in Python.

To export to HDF5 in SLEAP, go to **File** → **Export Analysis HDF5...** → **Current Video...**

See how to read HDF5 file [here](../learnings/export-analysis.md).

## CSV

Comma-separated value (CSV) files are the simplest and most portable format for tabular data, and can be opened in standard spreadsheet software like Microsoft Excel or Google Sheets.

While it does not allow for storing arbitrary metadata, it can be useful when you just need the pose trajectories and no other information such as the skeleton, provenance and others.

A number of behavioral analysis tools also support reading this format and it is particularly easy to manipulate.

To export to CSV in SLEAP, go to **File** → **Export Analysis CSV...** → **Current Video...**


You did it!

[*Next up:* I'm done SLEAPing, now what?](i-m-done-sleaping-now-what.md)