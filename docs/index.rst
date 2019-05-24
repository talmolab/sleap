.. image:: _static/supp_mov1-long_clip.gif
    :width: 500px

|

Social LEAP Estimates Animal Pose (sLEAP)
=========================================

**S**\ ocial **L**\ EAP **E**\ stimates **A**\ nimal **P**\ ose (**sLEAP**) is a framework for multi-animal
body part position estimation via deep learning. It is the successor to LEAP_. **sLEAP** is written entirely in
Python, supports multi-animal pose estimation, animal instance tracking, and a labeling\\training GUI that
supports active learning.

.. _LEAP: https://github.com/talmo/leap

.. _Installation:

Installation
------------

**sLEAP** is compatible with python versions 3.6 and above, with support for Windows and Linux. Mac OSX will probably
work as well, however, this is untested as of now.

Windows
-------

Since **sLEAP** has a number of complex binary dependencies (TensorFlow, Keras, OpenCV), it is recommended to use the
Anaconda_ python distribution to simplify installation.

Once Anaconda_ has been installed, go to start menu and type in *Anaconda*, which should bring up a menu entry
**Anaconda Prompt** which opens a command line with the base anaconda environment activated. One of the key
advantages to using `Anaconda Environments`_ is the ability to create separate Python installations (environments) for
different projects, mitigating issues of managing complex dependencies. To create a new conda environment for
**sLEAP** related development and use:

::

    (base) C:\> conda create -n sleap_env python=3.6 -c sleap sleap -y

Once the environment is finished installing, it can be activated using the following command:

::

    (base) C:\> conda activate sleap_env
    (sleap_env) C:\>

Any Python installation commands (:code:`conda install` or :code:`pip install`) issued after activating an
environment will only effect the environment. Thus it is important to make sure the environment is active when issuing
any commands that deal with Python on the command line.

**sLEAP** is now installed in the :code:`sleap_env` conda environment. With the environment active,
you can run the labeling GUI by entering the following command:

::

(sleap_env) C:\> sleap-label

.. _Anaconda: https://www.anaconda.com/distribution/
.. _Anaconda Environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Linux
-----

No linux conda packages are currently provided by the **sLEAP** channel. However, installing via :code:`pip` should not
be difficult on most Linux systems. The first step is to get a working version of tensorflow installed in your python
environment. Follow official directions for installing TensorFlow_ with GPU support. Once tensor flow is installed, simple
issue the following command to install **sLEAP**

.. _TensorFlow: https://www.tensorflow.org/install/gpu

::

    pip install git+https://github.com/murthylab/sleap.git

**sLEAP** is now installed you can run the labeling GUI by entering the following command:

::

> sleap-label

.. _sleap_package:
.. toctree::
    :caption: sLEAP Package
    :maxdepth: 1

    skeleton
    video
    instance
    dataset
    training
    inference
    gui

.. _Indices_and_Tables:

.. _Contributors:

Contributors
------------

* **Talmo Pereira**, Princeton Neuroscience Institute, Princeton University
* **Nat Tabris**, Princeton Neuroscience Institute, Princeton University
* **David Turner**, Research Computing, Princeton University

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`