.. _installation:

Installation
============

**SLEAP** is compatible with Python versions 3.6 and above, with support for Windows and Linux. Mac OS X works but without GPU support.

Windows
-------

Since **SLEAP** has a number of complex binary dependencies (TensorFlow, Keras, OpenCV), it is recommended to use the
Anaconda_ Python distribution to simplify installation. If you don't already have Anaconda installed, go to the Anaconda_ website and follow their installation instructions.

Once Anaconda_ has been installed, go to start menu and type in *Anaconda*, which should bring up a menu entry
**Anaconda Prompt** which opens a command line with the base anaconda environment activated. One of the key
advantages to using `Anaconda Environments`_ is the ability to create separate Python installations (environments) for
different projects, mitigating issues of managing complex dependencies. To create a new conda environment for
**SLEAP** related development and use:

::

    (base) C:\>  conda create -n sleap_env -c defaults -c sleap sleap=1.0.0 python=3.6 -y

Once the environment is finished installing, it can be activated using the following command:

::

    (base) C:\> conda activate sleap_env
    (sleap_env) C:\>

Any Python installation commands (:code:`conda install` or :code:`pip install`) issued after activating an
environment will only effect the environment. Thus it is important to make sure the environment is active when issuing
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

If you are on Linux and have a GPU supported by TensorFlow which you which to use, you should follow official directions for installing TensorFlow_ with GPU support. There is no TensorFlow GPU support on MacOS X.

.. _TensorFlow: https://www.tensorflow.org/install/gpu



You can then install SLEAP by running:

::

    pip install sleap==1.0.0

**SLEAP** is now installed you can run the labeling GUI by entering the following command:

::

    sleap-label

Developer Installation
----------------------

If you want to install a version of **SLEAP** which allows you to modify the code, see `these instructions`_ for the developer installation.

.. _these instructions: https://github.com/murthylab/sleap/wiki/Installation-methods
