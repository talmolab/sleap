.. _installation:

Installation
============

**SLEAP** is compatible with Python versions 3.6 and above, with support for Windows and Linux. Mac OS X works but without GPU support.

To find info on the latest version of SLEAP, check out `Releases <https://github.com/murthylab/sleap/releases>`_.

Quick Install
-------------

**Windows or Linux:** (requires Miniconda_ or Anaconda_)

::

    conda install -c sleap -n sleap sleap

**Mac or Colab:**

::

    pip install sleap

See the sections below for more detailed information.


.. _`conda install`:

Conda (Windows/Linux)
---------------------

Since **SLEAP** has a number of complex binary dependencies (TensorFlow, Keras, OpenCV), it is recommended to use the Anaconda_ Python distribution to simplify installation. To make things easier, we provide SLEAP conda packages that come with built-in GPU support (CUDA and CuDNN).

Windows
+++++++

.. note:: You can also use the Linux instructions below if you are comfortable with the command line.

If you don't already have Anaconda installed, go to the Anaconda_ website and follow the installation instructions.

Once Anaconda_ has been installed, go to start menu and type in *Anaconda*, which should bring up a menu entry
**Anaconda Prompt** which opens a command line with the base anaconda environment activated. One of the key
advantages to using `Anaconda Environments`_ is the ability to create separate Python installations (environments) for
different projects, mitigating issues of managing complex dependencies. To create a new conda environment for
**SLEAP** related development and use:

::

    (base) C:\>  conda create -y -n sleap -c defaults -c sleap sleap python=3.6 -y

Once the environment is finished installing, it can be activated using the following command:

::

    (base) C:\> conda activate sleap
    (sleap) C:\>

Any Python installation commands (``conda install`` or ``pip install``) issued after activating an
environment will only affect the active environment. Thus it is important to make sure the environment is active when issuing
any commands that deal with Python on the command line.

**SLEAP** is now installed in the ``sleap`` conda environment. With the environment active,
you can run the labeling GUI by entering the following command:

::

(sleap) C:\> sleap-label



Linux
+++++
If you are a bit more comfortable with the command line and can manage your environment yourself, we recommend using Miniconda_.

On Linux you can install it with:

::

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda

This will install ``conda`` to your home directory. To auto-start it when you open a new terminal so you can do ``conda`` commands more easily, run these lines:

::

    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init

Now you'll be ready to install SLEAP in a new environment called ``sleap``:

::

    conda create -y -n sleap -c sleap sleap


Mac OS X
++++++++

Currently we don't have a conda package for Mac OS X, but it is easy to install SLEAP via ``pip``.

First, follow the instructions above to install Anaconda or Miniconda. Then, run these commands to create a compatible environment:

::

    conda create -n sleap python=3.6 -y
    conda activate sleap

Next, simply ``pip`` install it:

::

    pip install sleap

.. _Anaconda: https://docs.anaconda.com/anaconda/install/
.. _Anaconda Environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html


.. _`pip install`:

Pip (Windows/Linux/Mac/Colab)
-----------------------------

We strongly recommend installing SLEAP into a ``conda`` environment with Python 3.6. If you know what you're doing and/or are running SLEAP in Colab or another virtual environment, simply run:

::

    pip install sleap


**SLEAP** is now installed, you can run the labeling GUI:

::

    sleap-label


Installation from source
------------------------

If you want to install a version of **SLEAP** which allows you to modify the code, follow these steps to check out the code and create a development environment:

::

    git clone https://github.com/murthylab/sleap.git
    cd sleap
    conda env create -n sleap_dev -f environment.yml
    conda activate sleap_dev

That's it! You can now make changes to the code and they will be reflected automatically when you run any SLEAP commands or scripts that import SLEAP.

To make sure everything's working, you can run the test suite:

::

    pytest tests


For more advanced users, if you'd like to install a particular branch of SLEAP without needing to edit it, you can use this syntax:

::

    pip install git+https://github.com/murthylab/sleap.git@develop

This will install the ``develop`` branch.

Uninstalling
------------

If you installed via ``conda``, just delete the environment:

::

    conda env remove -n sleap

Installed SLEAP in more than one environment? You can check the list of environments on your system with ``conda env list``.

If you installed via ``pip``:

::

    pip uninstall sleap



Troubleshooting
---------------

Running into installation issues? Open a new `GitHub Issue <https://github.com/murthylab/sleap/issues>`_ and let us know what you've tried so far.
