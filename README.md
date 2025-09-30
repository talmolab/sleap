[![CI](https://github.com/talmolab/sleap/workflows/CI/badge.svg?event=push&branch=develop)](https://github.com/talmolab/sleap/actions?query=workflow:CI)
[![Coverage](https://codecov.io/gh/talmolab/sleap/branch/develop/graph/badge.svg?token=oBmTlGIQRn)](https://codecov.io/gh/talmolab/sleap)
[![Documentation](https://img.shields.io/badge/Documentation-sleap.ai-lightgrey)](https://docs.sleap.ai)
[![Downloads](https://static.pepy.tech/personalized-badge/sleap?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20Downloads)](https://pepy.tech/project/sleap)
[![Conda Downloads](https://img.shields.io/conda/dn/sleap/sleap?label=Conda%20Downloads)](https://anaconda.org/sleap/sleap)
[![Stable version](https://img.shields.io/github/v/release/talmolab/sleap?label=stable)](https://github.com/talmolab/sleap/releases/)
[![Latest version](https://img.shields.io/github/v/release/talmolab/sleap?include_prereleases&label=latest)](https://github.com/talmolab/sleap/releases/)

# Social LEAP Estimates Animal Poses (SLEAP)

![SLEAP Demo](https://docs.sleap.ai/docs/_static/sleap_movie.gif)

**SLEAP** is an open source deep-learning based framework for multi-animal pose tracking [(Pereira et al., Nature Methods, 2022)](https://www.nature.com/articles/s41592-022-01426-1). It can be used to track any type or number of animals and includes an advanced labeling/training GUI for active learning and proofreading.

## Features

* Easy, one-line installation with support for all OSes
* Purpose-built GUI and human-in-the-loop workflow for rapidly labeling large datasets
* Single- and multi-animal pose estimation with *top-down* and *bottom-up* training strategies
* State-of-the-art pretrained and customizable neural network architectures that deliver *accurate predictions* with *very few* labels
* Fast training: 15 to 60 mins on a single GPU for a typical dataset
* Fast inference: up to 600+ FPS for batch, <10ms latency for realtime
* Support for remote training/inference workflow (for using SLEAP without GPUs)
* Flexible developer API for building integrated apps and customization

## Get some SLEAP

SLEAP is installed as a Python package. We strongly recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install SLEAP in its own environment.

You can find the latest version of SLEAP in the [Releases](https://github.com/talmolab/sleap/releases) page.

### Quick install

`conda` **(Windows/Linux/GPU)**:

```bash
conda create -y -n sleap -c conda-forge -c nvidia -c sleap/label/dev -c sleap -c anaconda sleap
```

`pip` **(any OS except Apple silicon)**:

```bash
pip install sleap[pypi]
```

See the docs for [full installation instructions](https://docs.sleap.ai/installation).

## Learn to SLEAP

- **Learn step-by-step**: [Tutorial](https://docs.sleap.ai/tutorial/overview)
- **Learn more advanced usage**: [Guides](https://docs.sleap.ai/how-to-guides/guides-overview/) and [Notebooks](https://docs.sleap.ai/notebooks/notebooks-overview/)
- **Learn by watching**: [ABL:AOC 2023 Workshop](https://www.youtube.com/watch?v=BfW-HgeDfMI) and [MIT CBMM Tutorial](https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking)
- **Learn by reading**: [Paper (Pereira et al., Nature Methods, 2022)](https://www.nature.com/articles/s41592-022-01426-1) and [Review on behavioral quantification (Pereira et al., Nature Neuroscience, 2020)](https://rdcu.be/caH3H)
- **Learn from others**: [Discussions on Github](https://github.com/talmolab/sleap/discussions)

## References

SLEAP is the successor to the single-animal pose estimation software [LEAP](https://github.com/talmo/leap) ([Pereira et al., Nature Methods, 2019](https://www.nature.com/articles/s41592-018-0234-5)).

If you use SLEAP in your research, please cite:

> T.D. Pereira, N. Tabris, A. Matsliah, D. M. Turner, J. Li, S. Ravindranath, E. S. Papadoyannis, E. Normand, D. S. Deutsch, Z. Y. Wang, G. C. McKenzie-Smith, C. C. Mitelut, M. D. Castro, J. D'Uva, M. Kislin, D. H. Sanes, S. D. Kocher, S. S-H, A. L. Falkner, J. W. Shaevitz, and M. Murthy. [Sleap: A deep learning system for multi-animal pose tracking](https://www.nature.com/articles/s41592-022-01426-1). *Nature Methods*, 19(4), 2022

**BibTeX:**

```bibtex
@ARTICLE{Pereira2022sleap,
   title={SLEAP: A deep learning system for multi-animal pose tracking},
   author={Pereira, Talmo D and 
      Tabris, Nathaniel and
      Matsliah, Arie and
      Turner, David M and
      Li, Junyu and
      Ravindranath, Shruthi and
      Papadoyannis, Eleni S and
      Normand, Edna and
      Deutsch, David S and
      Wang, Z. Yan and
      McKenzie-Smith, Grace C and
      Mitelut, Catalin C and
      Castro, Marielisa Diez and
      D'Uva, John and
      Kislin, Mikhail and
      Sanes, Dan H and
      Kocher, Sarah D and
      Samuel S-H and
      Falkner, Annegret L and
      Shaevitz, Joshua W and
      Murthy, Mala},
   journal={Nature Methods},
   volume={19},
   number={4},
   year={2022},
   publisher={Nature Publishing Group}
   }
}
```

## Contact

Follow [@talmop](https://twitter.com/talmop) on Twitter for news and updates!

**Technical issue with the software?**

1. Check the [Help page](https://docs.sleap.ai/help).
2. Ask the community via [discussions on Github](https://github.com/talmolab/sleap/discussions).
3. Search the [issues on GitHub](https://github.com/talmolab/sleap/issues) or open a new one.

**General inquiries?**
Reach out to talmo@salk.edu.

## Contributors

* **Talmo Pereira**, Salk Institute for Biological Studies
* **Liezl Maree**, Salk Institute for Biological Studies
* **Arlo Sheridan**, Salk Institute for Biological Studies
* **Arie Matsliah**, Princeton Neuroscience Institute, Princeton University
* **Nat Tabris**, Princeton Neuroscience Institute, Princeton University
* **David Turner**, Research Computing and Princeton Neuroscience Institute, Princeton University
* **Joshua Shaevitz**, Physics and Lewis-Sigler Institute, Princeton University
* **Mala Murthy**, Princeton Neuroscience Institute, Princeton University

SLEAP was created in the [Murthy](https://murthylab.princeton.edu) and [Shaevitz](https://shaevitzlab.princeton.edu) labs at the [Princeton Neuroscience Institute](https://pni.princeton.edu) at Princeton University.

SLEAP is currently being developed and maintained in the [Talmo Lab](https://talmolab.org) at the [Salk Institute for Biological Studies](https://salk.edu), in collaboration with the Murthy and Shaevitz labs at Princeton University.

This work was made possible through our funding sources, including:

* NIH BRAIN Initiative R01 NS104899
* Princeton Innovation Accelerator Fund

## License

SLEAP is released under a [Clear BSD License](https://raw.githubusercontent.com/talmolab/sleap/main/LICENSE) and is intended for research/academic use only. For commercial use, please contact: Laurie Tzodikov (Assistant Director, Office of Technology Licensing), Princeton University, 609-258-7256.

## Links

* [Documentation Homepage](https://docs.sleap.ai)
* [Overview](https://docs.sleap.ai/overview)
* [Installation](https://docs.sleap.ai/installation)
* [Tutorial](https://docs.sleap.ai/tutorial/overview/)
* [Guides](https://docs.sleap.ai/how-to-guides/guides-overview/)
* [Notebooks](https://docs.sleap.ai/notebooks/notebooks-overview/)
* [Developer API](https://docs.sleap.ai/api)
* [Help](https://docs.sleap.ai/help)