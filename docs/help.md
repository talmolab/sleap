# Help

Stuck? Can't get SLEAP to run? Crashing? Try the recommended tips below.


## Installation

### I can't get SLEAP to install!

Have you tried all of the steps in the {ref}`installation instructions <installation>`?

If so, please feel free to [open an issue](https://github.com/murthylab/sleap/issues) and tell us how you're trying to install it, what error messages you're getting and which operating system you're on.

### Can I install it on a computer without a GPU?

Yes! You can install SLEAP as you normally would using the `conda` or `pip`-based methods and the GPU support will be ignored.

### What if I already have CUDA set up on my system?

You can use the system CUDA installation by simply using the {ref}`pip <pip install>` installation method.

Note that you will need to use a version compatible with **TensorFlow 2.6+** (**CUDA Toolkit v11.3** and **cuDNN v8.2**).


## Usage

### How do I use SLEAP?

If you're new to pose tracking in general, check out [this talk](https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking) or our review in *[Nature Neuroscience](https://rdcu.be/caH3H)*.

If you're just new to SLEAP, we suggest starting with the {ref}`high-level overview` and then following the {ref}`tutorial`.

Once you get the hang of it, check out the {ref}`guides` for more detailed info.

(reencoding)=

### Does my data need to be in a particular format?

SLEAP supports a large number of formats, including all video formats and imported data from DeepLabCut and others.

Many types of video acquisition software, however, do not save videos in a format suitable for computer vision-based processing. A very common issue is that videos are not **reliably seekable**, meaning that you may not get the same data when trying to read a particular frame index. This is because many video formats are optimized for realtime and sequential playback and save space by reconstructing the image using data in adjacent frames. The consequence is that you may not get the exact same image depending on what the last frame you read was. Check out [this blog post](http://blog.loopbio.com/video-io-1-introduction.html) for more details.

If you think you may be affected by this issue, or just want to be safe, re-encode your videos using the following command:

```
ffmpeg -y -i "input.mp4" -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 "output.mp4"
```

Breaking down what this does:

- `-i "input.mp4"`: Specifies the path to the input file. Replace this with your video. Can be `.avi` or any other video format.
- `-c:v libx264`: Sets the video compression to use H264.
- `-pix_fmt yuv420p`: Necessary for playback on some systems.
- `-preset superfast`: Sets a number of parameters that enable reliable seeking.
- `-crf 23`: Sets the quality of the output video. Lower numbers are less lossy, but result in larger files. A CRF of 15 is nearly lossless, while 30 will be highly compressed.
- `"output.mp4"`: The name of the output video file (must end in `.mp4`).

If you don't have `ffmpeg` on your system, you can install it using `conda install ffmpeg` or by downloading it from the [official website](https://ffmpeg.org/download.html).

### I get strange results where the poses appear to be correct but shifted relative to the image.

This is most likely an issue with the video compression format. {ref}`See above <reencoding>`.

### How do I get predictions out?

See {ref}`export-analysis` and {ref}`sleap-convert`.

### What do I do with the output of SLEAP?

Check out the [Analysis examples](notebooks/Analysis_examples) notebooks.

## Getting more help

### I've found a bug or have another problem!

Feel free to [open an issue](https://github.com/murthylab/sleap/issues) and we'll get back to you as soon as we can!

### Can I just talk to someone?

SLEAP is a complex machine learning system intended for general use, so it's possible that we failed to consider the specifics of the situation in which you may be interested in using it with.

Feel free to reach out to us at `talmo@salk.edu` if you have a question that isn't covered here.