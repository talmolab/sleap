.. _installation:

Installation
============

**SLEAP** is compatible with Python versions 3.6 and above, with support for Windows and Linux. Mac OS X works but without GPU support.

GPU Support
-----------

**SLEAP** relies on `TensorFlow <https://www.tensorflow.org>`_ for training and inference. TensorFlow can use an NVIDIA GPU on Windows and Linux. Other GPUs—AMD, Intel, or older NVIDA GPUs on Macs—are not supported. For more details, see the `TensorFlow GPU support <https://www.tensorflow.org/install/gpu>`_ documentation.

.. note::
    It's possible you can run TensorFlow on AMD GPUs using `AMD ROCm <https://rocmdocs.amd.com/en/latest/Deep_learning/Deep-learning.html#tensorflow-installation>`_. We haven't tried this but if you're brave enough to try and you get it to work, let us know!

Without a supported GPU you'll still be able to use **SLEAP** although training on your local machine will be very, very slow; inference will be  slower than it would be with a GPU but may be tolerable.

If you don't have a supported GPU installed, we suggest using your local computer for labeling your dataset and then training models using `Google Colab <https://colab.research.google.com>`_ (free!) or an HPC cluster (if you have access to one). See our :ref:`guides` for more information about running **SLEAP** remotely.

Windows
-------

Since **SLEAP** has a number of complex binary dependencies (TensorFlow, Keras, OpenCV), it is recommended to use the Anaconda_ Python distribution to simplify installation. Anaconda will also install the NVIDIA GPU drivers which TensorFlow needs for running on the GPU.

If you don't already have Anaconda installed, go to the Anaconda_ website and follow their installation instructions.

Once Anaconda_ has been installed, go to start menu and type in *Anaconda*, which should bring up a menu entry
**Anaconda Prompt** which opens a command line with the base anaconda environment activated. One of the key
advantages to using `Anaconda Environments`_ is the ability to create separate Python installations (environments) for
different projects, mitigating issues of managing complex dependencies. To create a new conda environment for
**SLEAP** related development and use:

::

    (base) C:\>  conda create -n sleap_env -c defaults -c sleap sleap=1.1.2 python=3.6 -y

Once the environment is finished installing, it can be activated using the following command:

::

    (base) C:\> conda activate sleap_env
    (sleap_env) C:\>

Any Python installation commands (:code:`conda install` or :code:`pip install`) issued after activating an
environment will only affect the active environment. Thus it is important to make sure the environment is active when issuing
any commands that deal with Python on the command line.

**SLEAP** is now installed in the :code:`sleap_env` conda environment. With the environment active,
you can run the labeling GUI by entering the following command:

::

(sleap_env) C:\> sleap-label

.. _Anaconda: https://www.anaconda.com/distribution/
.. _Anaconda Environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Linux and MacOS X
-----------------

Currently we don't have an up-do-date conda package for Linux or MacOS X. It is easy to install SLEAP via :code:`pip` on Linux and MacOS X.

We recommend installing SLEAP into an environment with Python 3.6. If you are using conda, you can create an environment by running:

::

    conda create -n sleap_env python=3.6 -y
    conda activate sleap_env

If you are on Linux and have a GPU supported by TensorFlow, you should follow official directions for installing `TensorFlow with GPU support <https://www.tensorflow.org/install/gpu>`_.

.. note::
    There is no TensorFlow GPU support on MacOS X.

You can then install SLEAP by running:

::

    pip install sleap==1.1.2

**SLEAP** is now installed, you can run the labeling GUI:

::

    sleap-label

Developer Installation
----------------------

If you want to install a version of **SLEAP** which allows you to modify the code, see `these instructions`_ for the developer installation.

.. _these instructions: https://github.com/murthylab/sleap/wiki/Installation-methods
