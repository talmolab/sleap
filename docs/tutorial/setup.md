---
hide:
  - toc
---

# 1. Setup

SLEAP uses deep neural networks to learn how to predict poses from data. Training these models is **much** faster when using a GPU[^+] for acceleration.

[^+]: Graphics processing unit. This is a hardware component that parallelizes computations across thousands of cores, making them particularly effective for the algorithms used to train deep neural networks.

If you know you have a GPU on your machine or have a [Mac with Apple Silicon](https://support.apple.com/en-us/116943), you can install SLEAP locally and follow along this tutorial.

!!! tip "sleap-nn neural-network backend"
    The SLEAP GUI can be installed and used independently of the sleap-nn backend for **labeling**. However, for this tutorial it is important that you have sleap-nn installed with the correct **PyTorch and CUDA versions** according to your machine (ex. CPU or GPU).

    To check which PyTorch and CUDA versions you should have installed, see [here](https://nn.sleap.ai/dev/installation/).


## Install SLEAP locally

**See the [main SLEAP installation instructions](../installation.md) for detailed installation instructions.**

If you have either a Linux or Windows laptop with a GPU, or a [Mac with Apple Silicon](https://support.apple.com/en-us/116943), SLEAP will work natively with hardware acceleration.
<!-- 
1. Make sure that you have [Miniforge](https://github.com/conda-forge/miniforge) installed.

    You can also use Miniconda or vanilla Anaconda, but we recommend Miniforge since it
    is *much* faster to resolve dependencies, thus making the installation much faster.
    If you use Anaconda, replace `mamba` with `conda` in the commands below.
    
    To install Miniforge:

    === "Windows"

        Open a new PowerShell terminal (does not need to be admin) and enter:

        ```bash
        Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe"; Start-Process -FilePath "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe" -ArgumentList "/InstallationType=JustMe /RegisterPython=1 /S" -Wait; Remove-Item -Path "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe"
        ```

    === "Linux"

        Open a new terminal and enter:

        ```bash
        curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "~/Downloads/Miniforge3-Linux-x86_64.sh" && chmod +x "~/Downloads/Miniforge3-Linux-x86_64.sh" && "~/Downloads/Miniforge3-Linux-x86_64.sh" -b -p ~/miniforge3 && rm "~/Downloads/Miniforge3-Linux-x86_64.sh" && ~/miniforge3/bin/conda init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
        ```

    === "Mac (Apple Silicon)"

        Open a new terminal and enter:

        ```bash
        curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o "~/Downloads/Miniforge3-MacOSX-arm64.sh" && chmod +x "~/Downloads/Miniforge3-MacOSX-arm64.sh" && "~/Downloads/Miniforge3-MacOSX-arm64.sh" -b -p ~/miniforge3 && rm "~/Downloads/Miniforge3-MacOSX-arm64.sh" && ~/miniforge3/bin/conda init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
        ```

    === "Mac (Intel)"

        Open a new terminal and enter:

        ```bash
        curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -o "~/Downloads/Miniforge3-MacOSX-x86_64.sh" && chmod +x "~/Downloads/Miniforge3-MacOSX-x86_64.sh" && "~/Downloads/Miniforge3-MacOSX-x86_64.sh" -b -p ~/miniforge3 && rm "~/Downloads/Miniforge3-MacOSX-x86_64.sh" && ~/miniforge3/bin/conda init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
        ```

2. Install SLEAP in a new environment.

    === "Windows/Linux"

        ```bash
        mamba create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.3
        ```

    === "Mac"

        ```bash
        mamba create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.3.3
        ``` -->


[*Next up:* Importing data](importing-data.md)