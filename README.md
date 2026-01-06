[![CI](https://github.com/talmolab/sleap/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/talmolab/sleap/branch/develop/graph/badge.svg?token=oBmTlGIQRn)](https://codecov.io/gh/talmolab/sleap)
[![Documentation](https://img.shields.io/badge/Documentation-sleap.ai-lightgrey)](https://docs.sleap.ai)
[![Downloads](https://static.pepy.tech/personalized-badge/sleap?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20Downloads)](https://pepy.tech/project/sleap)
[![Conda Downloads](https://img.shields.io/conda/dn/sleap/sleap?label=Conda%20Downloads)](https://anaconda.org/sleap/sleap)
[![Stable version](https://img.shields.io/github/v/release/talmolab/sleap?label=stable)](https://github.com/talmolab/sleap/releases/)
[![Latest version](https://img.shields.io/github/v/release/talmolab/sleap?include_prereleases&label=latest)](https://github.com/talmolab/sleap/releases/)

# Social LEAP Estimates Animal Poses (SLEAP)

![SLEAP Demo](https://raw.githubusercontent.com/talmolab/sleap/develop/docs/assets/images/sleap_movie.gif)

**SLEAP** is an open-source deep-learning based framework for multi-animal pose tracking [(Pereira et al., Nature Methods, 2022)](https://www.nature.com/articles/s41592-022-01426-1). It can be used to track any type or number of animals and includes an advanced labeling/training GUI for active learning and proofreading.

## Features

* Easy, one-line installation with support for all OSes
* Purpose-built GUI and human-in-the-loop workflow for rapidly labeling large datasets
* Single- and multi-animal pose estimation with *top-down* and *bottom-up* training strategies
* Customizable neural network architectures that deliver *accurate predictions* with *very few* labels
* Fast training: 15 to 60 mins on a single GPU for a typical dataset
* Fast inference: up to 600+ FPS for batch, <10ms latency for realtime
* Support for remote training/inference workflow (for using SLEAP without GPUs)
* Flexible developer API for building integrated apps and customization
* Two independent backends— [`sleap-nn`](https://nn.sleap.ai) and [`sleap-io`](https://io.sleap.ai) for training/inference pipelines & handling SLEAP files respectively

## Get some SLEAP

SLEAP is installed as a Python package. We strongly recommend using [uv](https://docs.astral.sh/uv/) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to install SLEAP in its own environment.

You can find the latest version of SLEAP in the [Releases](https://github.com/talmolab/sleap/releases) page.

### Quick install

> **Python 3.14 is not yet supported**
>
> SLEAP currently supports **Python 3.11, 3.12, and 3.13**.  
> **Python 3.14 is not yet tested or supported.**  
> By default, `uv` will use your system-installed Python.  
> If you have Python 3.14 installed, you must specify the Python version (≤3.13) in the install command.  
>
> For example:
>
> ```bash
> uv tool install --python 3.13 "sleap[nn]"  ...
> ```
> Replace `...` with the rest of your install command as needed.

**`uv tool install` (any OS):**

First, install [`uv`](https://github.com/astral-sh/uv) if you haven't already:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install SLEAP:

```bash
# Windows/Linux CUDA 12.8
uv tool install "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple

# macOS / CPU-only
uv tool install "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
```

Run the SLEAP GUI after installation:

```bash
sleap
```

See the docs for [full installation instructions](https://docs.sleap.ai/latest/installation).

## Learn to SLEAP

- **Learn step-by-step:** [Tutorial](https://docs.sleap.ai/latest/tutorial/overview)
- **Learn more advanced usage:** [Guides](https://docs.sleap.ai/latest/guides/guides-overview/) and [Notebooks](https://docs.sleap.ai/latest/notebooks/notebooks-overview/)
- **Learn by watching:** [COSYNE 2024 Tutorial (Part 1)](https://youtu.be/R5PRhkhAve0), [COSYNE 2024 Tutorial (Part 2)](https://youtu.be/Z64v-vp-Jvo), [ABL:AOC 2023 Workshop](https://www.youtube.com/watch?v=BfW-HgeDfMI), and [MIT CBMM Tutorial](https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking)
- **Learn by reading:** [Paper (Pereira et al., Nature Methods, 2022)](https://www.nature.com/articles/s41592-022-01426-1) and [Review on behavioral quantification (Pereira et al., Nature Neuroscience, 2020)](https://rdcu.be/caH3H)
- **Learn from others:** [Discussions on Github](https://github.com/talmolab/sleap/discussions)

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


**Technical issue with the software?**

1. Check the [Help page](https://docs.sleap.ai/latest/help).
2. Ask the community via [discussions on Github](https://github.com/talmolab/sleap/discussions).
3. Search the [issues on GitHub](https://github.com/talmolab/sleap/issues) or open a new one.

**General inquiries?**
Reach out to [talmo@salk.edu](mailto:talmo@salk.edu).

## Contributors

* **Talmo Pereira**, Salk Institute for Biological Studies
* **Divya Murali**, Salk Institute for Biological Studies
* **Elizabeth Berrigan**, Salk Institute for Biological Studies
* **Amick Licup**, Salk Institute for Biological Studies
* **Andrew Park**, Salk Institute for Biological Studies
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
* NIH BRAIN Initiative RF1 MH132653

## License

SLEAP is released under a [Clear BSD License](https://raw.githubusercontent.com/talmolab/sleap/main/LICENSE) and is intended for research/academic use only. For commercial use, please contact: **Laurie Tzodikov (Assistant Director, Office of Technology Licensing), Princeton University, 609-258-7256**.

## Links

* [Documentation Homepage](https://docs.sleap.ai)
* [Overview](https://docs.sleap.ai/latest/overview)
* [Installation](https://docs.sleap.ai/latest/installation)
* [Tutorial](https://docs.sleap.ai/latest/tutorial/overview/)
* [Guides](https://docs.sleap.ai/latest/guides/guides-overview/)
* [Notebooks](https://docs.sleap.ai/latest/notebooks/notebooks-overview/)
* [Developer API](https://docs.sleap.ai/latest/api)
* [Help](https://docs.sleap.ai/latest/help)
