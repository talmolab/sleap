|CI| |Coverage| |Documentation| |Downloads| |Stable version| |Latest version|

.. |CI| image:: 
   https://github.com/talmolab/sleap/workflows/CI/badge.svg?event=push&branch=develop
   :target: https://github.com/talmolab/sleap/actions?query=workflow:CI
   :alt: Continuous integration status

.. |Coverage| image::
   https://codecov.io/gh/murthylab/sleap/branch/develop/graph/badge.svg?token=YWQYBN6820
   :target: https://codecov.io/gh/talmolab/sleap
   :alt: Coverage

.. |Documentation| image:: 
   https://img.shields.io/github/workflow/status/murthylab/sleap/Build%20website?label=Documentation
   :target: https://sleap.ai
   :alt: Documentation
  
.. |Downloads| image::
   https://static.pepy.tech/personalized-badge/sleap?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/sleap
   :alt: Downloads

.. |Stable version| image:: https://img.shields.io/github/v/release/murthylab/sleap?label=stable
   :target: https://github.com/talmolab/sleap/releases/
   :alt: Stable version

.. |Latest version| image:: https://img.shields.io/github/v/release/murthylab/sleap?include_prereleases&label=latest
   :target: https://github.com/talmolab/sleap/releases/
   :alt: Latest version


.. start-inclusion-marker-do-not-remove


Social LEAP Estimates Animal Poses (SLEAP)
==========================================

.. image:: https://sleap.ai/docs/_static/sleap_movie.gif
    :width: 600px

**SLEAP** is an open source deep-learning based framework for multi-animal pose tracking. It can be used to track any type or number of animals and includes an advanced labeling/training GUI for active learning and proofreading.


Features
--------
* Easy, one-line installation with support for all OSes
* Purpose-built GUI and human-in-the-loop workflow for rapidly labeling large datasets
* Single- and multi-animal pose estimation with *top-down* and *bottom-up* training strategies
* State-of-the-art pretrained and customizable neural network architectures that deliver *accurate predictions* with *very few* labels
* Fast training: 15 to 60 mins on a single GPU for a typical dataset
* Fast inference: up to 600+ FPS for batch, <10ms latency for realtime
* Support for remote training/inference workflow (for using SLEAP without GPUs)
* Flexible developer API for building integrated apps and customization


Get some SLEAP
--------------
SLEAP is installed as a Python package. We strongly recommend using `Miniconda <https://https://docs.conda.io/en/latest/miniconda.html>`_ to install SLEAP in its own environment.

You can find the latest version of SLEAP in the `Releases <https://github.com/talmolab/sleap/releases>`_ page.

Quick install
^^^^^^^^^^^^^
`conda` **(Windows/Linux/GPU)**:

.. code-block:: bash

    conda create -y -n sleap -c sleap -c nvidia -c conda-forge sleap


`pip` **(any OS)**:

.. code-block:: bash

    pip install sleap


See the docs for `full installation instructions <https://sleap.ai/installation.html>`_.

Learn to SLEAP
--------------
- **Learn step-by-step**: `Tutorial <https://sleap.ai/tutorials/tutorial.html>`_
- **Learn more advanced usage**: `Guides <https://sleap.ai/guides/>`_ and `Notebooks <https://sleap.ai/notebooks/>`_
- **Learn by watching**: `MIT CBMM Tutorial <https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking>`_
- **Learn by reading**: `Paper (Pereira et al., bioRxiv, 2020) <https://doi.org/10.1101/2020.08.31.276246>`_ and `Review on behavioral quantification (Pereira et al., Nature Neuroscience, 2020) <https://rdcu.be/caH3H>`_


References
-----------
SLEAP is the successor to the single-animal pose estimation software `LEAP <https://github.com/talmo/leap>`_ (`Pereira et al., Nature Methods, 2019 <https://www.nature.com/articles/s41592-018-0234-5>`_).

If you use SLEAP in your research, please cite:

    Pereira, Talmo D., Nathaniel Tabris, Junyu Li, Shruthi Ravindranath, Eleni S. Papadoyannis, Z. Yan Wang, David M. Turner, et al. 2020. "SLEAP: Multi-Animal Pose Tracking." *bioRxiv*. https://doi.org/10.1101/2020.08.31.276246.


**BibTeX:**

.. code-block:: tex

   @ARTICLE{Pereira2020-tt,
   title    = "{SLEAP}: Multi-animal pose tracking",
   author   = "Pereira, Talmo D and Tabris, Nathaniel and Li, Junyu and
               Ravindranath, Shruthi and Papadoyannis, Eleni S and Yan Wang, Z
               and Turner, David M and McKenzie-Smith, Grace and Kocher, Sarah D
               and Falkner, Annegret Lea and Shaevitz, Joshua W and Murthy, Mala",
   journal  = "bioRxiv",
   pages    = "2020.08.31.276246",
   month    =  sep,
   year     =  2020,
   language = "en"
   }


Contact
-------

Follow `@talmop <https://twitter.com/talmop>`_ on Twitter for news and updates!

**Technical issue with the software?**

1. Check the `Help page <https://sleap.ai/help.html>`_.
2. Search the `issues on GitHub <https://github.com/talmolab/sleap/issues>`_ or open a new one.


**General inquiries?**
Reach out to `talmo@salk.edu`.

.. _Contributors:

Contributors
------------

* **Talmo Pereira**, Salk Institute for Biological Studies
* **Liezl Maree**, Salk Institute for Biological Studies
* **Arie Matsliah**, Princeton Neuroscience Institute, Princeton University
* **Nat Tabris**, Princeton Neuroscience Institute, Princeton University
* **David Turner**, Research Computing and Princeton Neuroscience Institute, Princeton University
* **Joshua Shaevitz**, Physics and Lewis-Sigler Institute, Princeton University
* **Mala Murthy**, Princeton Neuroscience Institute, Princeton University

SLEAP was created in the `Murthy <https://murthylab.princeton.edu>`_ and `Shaevitz <https://shaevitzlab.princeton.edu>`_ labs at the `Princeton Neuroscience Institute <https://pni.princeton.edu>`_ at Princeton University.

SLEAP is currently being developed and maintained in the `Talmo Lab <https://talmolab.org>`_ at the `Salk Institute for Biological Studies <https://salk.edu>`_, in collaboration with the Murthy and Shaevitz labs at Princeton University.

This work was made possible through our funding sources, including:

* NIH BRAIN Initiative R01 NS104899
* Princeton Innovation Accelerator Fund


License
-------
SLEAP is released under a `Clear BSD License <https://raw.githubusercontent.com/talmolab/sleap/main/LICENSE>`_ and is intended for research/academic use only. For commercial use, please contact: Laurie Tzodikov (Assistant Director, Office of Technology Licensing), Princeton University, 609-258-7256.


.. end-inclusion-marker-do-not-remove

Links
------
* `Documentation Homepage <https://sleap.ai>`_
* `Overview <https://sleap.ai/overview.html>`_
* `Installation <https://sleap.ai/installation.html>`_
* `Tutorial <https://sleap.ai/tutorials/tutorial.html>`_
* `Guides <https://sleap.ai/guides/index.html>`_
* `Notebooks <https://sleap.ai/notebooks/index.html>`_
* `Developer API <https://sleap.ai/api.html>`_
* `Help <https://sleap.ai/help.html>`_
