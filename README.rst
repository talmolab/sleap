Social LEAP Estimates Animal Pose (SLEAP)
=========================================

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


.. image:: https://sleap.ai/docs/_static/sleap_movie.gif
    :width: 600px

|

**S**\ ocial **L**\ EAP **E**\ stimates **A**\ nimal **P**\ ose (**SLEAP**) is a framework for multi-animal
body part position estimation via deep learning. It is the successor to LEAP_ (`Pereira et al., 2019`_). **SLEAP** is written entirely in
Python, supports multi-animal pose estimation, animal instance tracking, and comes with a labeling/training GUI that
supports active learning.

If you use **SLEAP** in your research, please cite:

    Talmo D. Pereira, Nathaniel Tabris, Junyu Li, Shruthi Ravindranath, Eleni S. Papadoyannis, Z. Yan Wang, David M. Turner, et al. 2020. "SLEAP: Multi-Animal Pose Tracking." *bioRxiv*. https://doi.org/10.1101/2020.08.31.276246.

SLEAP is released under a `Clear BSD License`_ and is intended for research/academic use only. For commercial use, please contact: Laurie Tzodikov (Assistant Director, Office of Technology Licensing), Princeton University, 609-258-7256.

.. _LEAP: https://github.com/talmo/leap
.. _Clear BSD License: https://raw.githubusercontent.com/murthylab/sleap/master/LICENSE
.. _Pereira et al., 2019: https://www.nature.com/articles/s41592-018-0234-5
.. _sleap.ai: https://sleap.ai
