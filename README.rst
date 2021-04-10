|CI| |Coverage| |Documentation| |Downloads| |Stable version| |Latest version|

.. |CI| image:: 
   https://github.com/murthylab/sleap/workflows/CI/badge.svg?event=push&branch=develop
   :target: https://github.com/murthylab/sleap/actions?query=workflow:CI
   :alt: Continuous integration status

.. |Coverage| image::
   https://codecov.io/gh/murthylab/sleap/branch/tf23/graph/badge.svg?token=YWQYBN6820
   :target: https://codecov.io/gh/murthylab/sleap
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
   :target: https://GitHub.com/murthylab/sleap/releases/
   :alt: Stable version

.. |Latest version| image:: https://img.shields.io/github/v/release/murthylab/sleap?include_prereleases&label=latest
   :target: https://GitHub.com/murthylab/sleap/releases/
   :alt: Latest version


.. start-inclusion-marker-do-not-remove


**SLEAP** - Social LEAP Estimates Animal Poses
==============================================

.. image:: https://sleap.ai/docs/_static/sleap_movie.gif
    :width: 600px

**SLEAP** is an open source deep-learning based framework for estimating positions of animal body parts. It supports *multi-animal pose estimation* and *tracking*, and includes an advanced labeling/training GUI for active learning and proofreading.

SLEAP is written in Python and uses TensorFlow 2 for machine learning and Qt/PySide2 for graphical user interface. SLEAP is the successor to `LEAP <https://github.com/talmo/leap>`_ (`Pereira et al., Nature Methods, 2019 <https://www.nature.com/articles/s41592-018-0234-5>`_).


Features
------------

* Purpose-built GUI and human-in-the-loop workflow for rapidly labeling large datasets
* Multi-animal pose estimation with *top-down* and *bottom-up* training strategies
* State-of-the-art pretrained and customizable neural network architectures that deliver *accurate predictions* with *very few* labels
* Fast training: 15 to 60 mins on a single GPU for a typical dataset
* Fast inference: 400+ FPS for batch, 10ms latency for realtime
* Support for remote training/inference workflow (for using SLEAP without GPUs)
* Flexible developer API for building integrated apps and customization


Getting started
----------------

To get started with SLEAP, head over to the `Documentation <https://sleap.ai>`_ where you'll find tutorials, guides and example notebooks.

To learn more about the technical side of SLEAP and multi-animal pose tracking, check out our `preprint on bioRxiv <https://doi.org/10.1101/2020.08.31.276246>`_ or watch the `tutorial on SLEAP <https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking>`_. For a more general introduction to the field of quantitative animal behavior, check out our `review in Nature Neuroscience <https://rdcu.be/caH3H>`_.

You can find the latest version of SLEAP in the `Releases <https://github.com/murthylab/sleap/releases>`_ page.


References
-----------
If you use **SLEAP** in your research, please cite:

    Talmo D. Pereira, Nathaniel Tabris, Junyu Li, Shruthi Ravindranath, Eleni S. Papadoyannis, Z. Yan Wang, David M. Turner, et al. 2020. "SLEAP: Multi-Animal Pose Tracking." *bioRxiv*. https://doi.org/10.1101/2020.08.31.276246.

License
-------
SLEAP is released under a `Clear BSD License <https://raw.githubusercontent.com/murthylab/sleap/main/LICENSE>`_ and is intended for research/academic use only. For commercial use, please contact: Laurie Tzodikov (Assistant Director, Office of Technology Licensing), Princeton University, 609-258-7256.

Contact
-------

Follow `@MurthyLab <https://twitter.com/MurthyLab>`_ on Twitter for news and updates!

**Technical issue with the software?** `Open an issue on GitHub. <https://github.com/murthylab/sleap/issues>`_

**Press inquiries? Interested in using SLEAP in a commercial application?** Reach out at `sleap@princeton.edu`_.

.. _sleap@princeton.edu: sleap@princeton.edu


.. _Contributors:

Contributors
------------

* **Talmo Pereira**, Princeton Neuroscience Institute, Princeton University
* **Arie Matsliah**, Princeton Neuroscience Institute, Princeton University
* **Nat Tabris**, Princeton Neuroscience Institute, Princeton University
* **David Turner**, Research Computing and Princeton Neuroscience Institute, Princeton University
* **Joshua Shaevitz**, Physics and Lewis-Sigler Institute, Princeton University
* **Mala Murthy**, Princeton Neuroscience Institute, Princeton University

SLEAP is developed in the `Murthy <https://murthylab.princeton.edu>`_ and `Shaevitz <https://shaevitzlab.princeton.edu>`_ labs at the `Princeton Neuroscience Institute <https://pni.princeton.edu>`_ at Princeton University. This work was made possible through our funding sources, including: NIH BRAIN Initiative R01 NS104899 and Princeton Innovation Accelerator Fund.

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

