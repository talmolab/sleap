# Social LEAP Estimates Animal Poses (SLEAP)
![SLEAP Logo](assets/images/sleap_movie.gif)

SLEAP is an open-source deep-learning based framework for multi-animal pose tracking ([Pereira et al., Nature Methods, 2022](https://www.nature.com/articles/s41592-022-01426-1)). It can be used to track any type or number of animals and includes an advanced labeling/training GUI for active learning and proofreading.


!!! warning "Documentation for New SLEAP Versions"
    This documentation is for the **latest version of SLEAP**.  
    If you are using **SLEAP version 1.4.1 or earlier**, please visit the [legacy documentation](http://legacy.sleap.ai).


!!! tip "New in SLEAP 1.5"
    Check out our [What's New](whats-new.md) page to learn about the latest features including UV-based installation, PyTorch backend, and new standalone libraries!


## Features

- Easy, one-line installation with support for all OSes

- Purpose-built GUI and human-in-the-loop workflow for rapidly labeling large datasets

- Single- and multi-animal pose estimation with *top-down* and *bottom-up* training strategies

- Customizable neural network architectures that deliver *accurate predictions* with *very few* labels

- Fast training: 15 to 60 mins on a single GPU for a typical dataset

- Fast inference: up to 600+ FPS for batch, <10ms latency for realtime

- Support for remote training/inference workflow (for using SLEAP without GPUs)

- Flexible developer API for building integrated apps and customization

- Two independent backends-- [`sleap-nn`](https://nn.sleap.ai) and [`sleap-io`](https://io.sleap.ai) for training/inference pipelines & handling SLEAP files respectively

<!-- # TODO: Update training time taken DS -->

!!! tip "`sleap-nn` Backend"
    The SLEAP GUI can be installed and used independently of the `sleap-nn` backend for **labeling**. However, for training and inference workflows, it is important that you have sleap-nn installed with the correct **PyTorch and CUDA versions** according to your machine (ex. CPU or GPU).
    
    Learn more about `sleap-nn` [here](https://nn.sleap.ai).

!!! tip "`sleap-io` Backend"
    For working with SLEAP files **directly from a CLI**, it is best to use `sleap-io`.
    
    Learn more about `sleap-io` [here](https://io.sleap.ai).

## Get some SLEAP

SLEAP is installed as a Python package. We strongly recommend using [uv](https://docs.astral.sh/uv/) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to install SLEAP in its own environment.

You can find the latest version of SLEAP in the [Releases](https://github.com/talmolab/sleap/releases) page.

### Quick start

!!! tip "Sample with `uvx`"
    Note that opening SLEAP w/ `uvx` will **not** install SLEAP onto your system, it will only **'invoke'** SLEAP.

**`uvx` (any OS):**

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uvx --from "sleap[nn]" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 sleap-label
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--extra-index-url` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "macOS/CPU Only"
    ```bash
    uvx --from "sleap[nn]" sleap-label
    ```

=== "SLEAP GUI Only"
    ```bash
    uvx --from "sleap" sleap-label
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

**`pip` (any OS)**:

```
pip install sleap
```

See the docs for [full installation instructions](installation.md).


## Learn to SLEAP

- **Learn step-by-step: [Tutorial](tutorial/overview.md)**
- **Learn more advanced usage: [Guides](how-to-guides/guides-overview.md) and [Notebooks](notebooks/notebooks-overview.md)**
- **Learn by watching: [ABL:AOC 2023 Workshop](https://www.youtube.com/watch?v=BfW-HgeDfMI) and [MIT CBMM Tutorial](https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking)**
- **Learn by reading: [Paper (Pereira et al., Nature Methods, 2022)](https://www.nature.com/articles/s41592-022-01426-1) and [Review on behavioral quantification (Pereira et al., Nature Neuroscience, 2020)](https://rdcu.be/caH3H)**
- **Learn from others: [Discussions on Github](https://github.com/talmolab/sleap/discussions)**

## References

SLEAP is the successor to the single-animal pose estimation software [LEAP](https://github.com/talmo/leap) ([Pereira et al., Nature Methods, 2019](https://www.nature.com/articles/s41592-018-0234-5)).

If you use SLEAP in your research, please cite:



    
    T.D. Pereira, N. Tabris, A. Matsliah, D. M. Turner, J. Li, S. Ravindranath, E. S. Papadoyannis, E. Normand, D. S. Deutsch, Z. Y. Wang, G. C. McKenzie-Smith, C. C. Mitelut, M. D. Castro, J. D’Uva, M. Kislin, D. H. Sanes, S. D. Kocher, S. S-H, A. L. Falkner, J. W. Shaevitz, and M. Murthy. Sleap: A deep learning system for multi-animal pose tracking. Nature Methods, 19(4), 2022


## BibTeX


    
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
    
## Contact

Follow [@talmop](https://x.com/talmop) on [X](https://x.com) for news and updates!

**Technical issue with the software?**

1. Check the [Help page](help).
2. Ask the community via [discussions on Github](https://github.com/talmolab/sleap/discussions).
3. Search the [issues on GitHub](https://github.com/talmolab/sleap/issues) or open a new one.

**General inquiries?**
Reach out to [talmo@salk.edu](mailto:talmo@salk.edu).

## Contributors

- **Talmo Pereira**, Salk Institute for Biological Studies
- **Liezl Maree**, Salk Institute for Biological Studies
- **Arlo Sheridan**, Salk Institute for Biological Studies
- **Arie Matsliah**, Princeton Neuroscience Institute, Princeton University
- **Nat Tabris**, Princeton Neuroscience Institute, Princeton University
- **David Turner**, Research Computing and Princeton Neuroscience Institute, Princeton University
- **Joshua Shaevitz**, Physics and Lewis-Sigler Institute, Princeton University
- **Mala Murthy**, Princeton Neuroscience Institute, Princeton University

SLEAP was created in the [Murthy](https://murthylab.princeton.edu) and [Shaevitz](https://shaevitzlab.princeton.edu) labs at the [Princeton Neuroscience Institute](https://pni.princeton.edu) at Princeton University.

SLEAP is currently being developed and maintained in the [Talmo Lab](https://talmolab.org) at the [Salk Institute for Biological Studies](https://salk.edu), in collaboration with the Murthy and Shaevitz labs at Princeton University.

This work was made possible through our funding sources, including:

- NIH BRAIN Initiative R01 NS104899
- Princeton Innovation Accelerator Fund

## License

SLEAP is released under a [Clear BSD License](https://raw.githubusercontent.com/talmolab/sleap/main/LICENSE) and is intended for research/academic use only. For commercial use, please contact: **Laurie Tzodikov (Assistant Director, Office of Technology Licensing), Princeton University, 609-258-7256**.