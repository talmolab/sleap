# 9. I'm done SLEAPing, now what?

At the end of the day, the outputs of SLEAP are pose coordinates. By itself, this representation only tells you where body parts are located in each frame of a video, so what do we do with it?

Some common use cases for pose are:

1. **Behavior segmentation**: where pose trajectories are used to predict the occurrence of specific behaviors at each frame. This is also known as *behavior classification*, *behavior scoring*, or, in the computer vision field, <a href="https://en.wikipedia.org/wiki/Activity_recognition#Vision-based_activity_recognition" target="_blank">action recognition  </a>.
2. **Kinematic analysis**: where pose trajectories themselves are used to characterize fine-grained movement patterning, such as in the study of <a href="https://neuromechanicslab.emory.edu/documents/publications-docs/TingChielbook15BiomechanicalAffordances.pdf" target="_blank">biomechanics  </a> or the <a href="https://www.cell.com/neuron/pdf/S0896-6273(17)30463-4.pdf" target="_blank">neural control of movement  </a>.

Below we list [some of the tools](#integrations) that work with SLEAP to enable these downstream applications and more.

For more reading on this topic, check out the [last section on this page](#literature).

We recommend that you finish off the tutorial by [checking out the notebooks](#notebooks) below.

## Notebooks

- [Using exported data](../notebooks/SLEAP_Tutorial_at_Cosyne_2024_Using_exported_data.ipynb) <a href="https://colab.research.google.com/github/talmolab/sleap/blob/develop/docs/notebooks/SLEAP_Tutorial_at_Cosyne_2024_Using_exported_data.ipynb" target="_blank">![](https://colab.research.google.com/assets/colab-badge.svg)</a> → In this notebook, we demonstrate how to load and work with the different exported file formats from the rest of the tutorial.

- [Analysis examples](../notebooks/Analysis_examples.ipynb)  </a> <a href="https://colab.research.google.com/github/talmolab/sleap/blob/develop/docs/notebooks/Analysis_examples.ipynb" target="_blank">![](https://colab.research.google.com/assets/colab-badge.svg)</a> → In this notebook, we demonstrate how to conduct basic kinematic and behavioral analyses yourself in Python.

- [Using idTracker.ai with SLEAP](../notebooks/sleap_io_idtracker_IDs.ipynb) <a href="https://colab.research.google.com/github/talmolab/sleap/blob/develop/docs/notebooks/sleap_io_idtracker_IDs.ipynb" target="_blank">![](https://colab.research.google.com/assets/colab-badge.svg)</a> → In this notebook, we demonstrate how to use identities provided by idTracker to assign tracks to SLEAP poses.

## Integrations

SLEAP works with a number of different tools and software in the ecosystem of behavioral analysis. Most of these work with [exported SLEAP data](exporting-the-results.md) or models.

### Behavior segmentation

- <a href="https://keypoint-moseq.readthedocs.io/en/latest/" target="_blank">Keypoint-MoSeq  </a>: Unsupervised behavior segmentation using autoregressive models and switching linear dynamical systems.

- <a href="https://github.com/YttriLab/A-SOID" target="_blank">A-SOiD  </a>: Semi-supervised behavior segmentation with a local web app GUI.

- <a href="https://github.com/YttriLab/B-SOID" target="_blank">B-SOiD  </a>: Supervised behavior segmentation with clustering initialization.

- <a href="https://simba-uw-tf-dev.readthedocs.io/en/latest/" target="_blank">SimBA  </a>: Supervised behavior segmentation with random forests.

- <a href="https://github.com/DeNardoLab/BehaviorDEPOT" target="_blank">BehaviorDEPOT  </a>

- <a href="https://github.com/mlfpm/deepof" target="_blank">DeepOF  </a>

### Realtime

- <a href="https://github.com/SchwarzNeuroconLab/DeepLabStream" target="_blank">DeepLabStream  </a>: Standalone closed-loop app with SLEAP support.

### Utility libraries

- <a href="https://io.sleap.ai/" target="_blank">sleap-io  </a>: Standalone pose I/O utilities.

- <a href="https://nwb-guide.readthedocs.io/" target="_blank">NWB GUIDE  </a>: GUI-based NWB conversion.

- <a href="https://github.com/rly/ndx-pose" target="_blank">ndx-pose  </a>: NWB extension for pose data.

- <a href="https://movement.neuroinformatics.dev/" target="_blank">movement  </a>: General purpose utilities downstream of pose tracking.

## Literature

Here we list some reviews and other papers to check out on the use of pose for behavior, neuroscience, and more.

### Behavioral quantification and ethology

Datta et al., *Neuron* (2019). <a href="https://pubmed.ncbi.nlm.nih.gov/31600508/" target="_blank">Computational Neuroethology: A Call to Action.</a>

Pereira et al., *Nature Neuroscience* (2020). <a href="https://rdcu.be/caH3H" target="_blank">Quantifying behavior to understand the brain.</a>

Marshall et al., *Current Opinion in Neurobiology* (2022). <a href="https://www.sciencedirect.com/science/article/abs/pii/S0959438822000071" rel="noopener noreferrer" target="_blank">Leaving flatland: Advances in 3D behavioral measurement.</a>

Kennedy, *Current Opinion in Neurobiology* (2022). <a href="https://www.sciencedirect.com/science/article/abs/pii/S0959438822000435" rel="noopener noreferrer" target="_blank">The what, how, and why of naturalistic behavior.</a>

Luxem et al., *eLife* (2023). <a href="https://elifesciences.org/articles/79305" target="_blank">Open-source tools for behavioral video analysis: Setup, methods, and best practices.</a>

### Ecology

Bertram et al., *Biological Reviews* (2022). <a href="https://onlinelibrary.wiley.com/doi/full/10.1111/brv.12844" target="_blank">Frontiers in quantifying wildlife behavioural responses to chemical pollution.</a>

Borowiec et al., *Methods in Ecology and Evolution* (2022). <a href="https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13901" target="_blank">Deep learning as a tool for ecology and evolution.</a>

Couzin and Heins, *Trends in Ecology and Evolution* (2023). <a href="https://pubmed.ncbi.nlm.nih.gov/36509561/" target="_blank">Emerging technologies for behavioral research in changing environments.</a>

### Pain

Sadler et al., *Nature Reviews Neuroscience* (2021). <a href="https://www.nature.com/articles/s41583-021-00536-7" target="_blank">Innovations and advances in modelling and measuring pain in animals.</a>

Jhumka and Abdus-Saboor, *Current Opinion in Neurobiology* (2022). <a href="https://www.sciencedirect.com/science/article/pii/S0959438822000927" rel="noopener noreferrer" target="_blank">Next generation behavioral sequencing for advancing pain quantification.</a>

Bohic et al., *Neuron* (2023). <a href="https://www.sciencedirect.com/science/article/abs/pii/S0896627323004646" rel="noopener noreferrer" target="_blank">Mapping the neuroethological signatures of pain, analgesia, and recovery in mice.</a>

### Home cage monitoring

Grieco et al., *Frontiers in Behavioral Neuroscience* (2021). <a href="https://www.frontiersin.org/articles/10.3389/fnbeh.2021.735387/full" target="_blank">Measuring Behavior in the Home Cage: Study Design, Applications, Challenges, and Perspectives.</a>

Klein et al., *Frontiers in Behavioral Neuroscience*. (2022). <a href="https://www.frontiersin.org/articles/10.3389/fnbeh.2022.877323/full" target="_blank">Measuring Locomotor Activity and Behavioral Aspects of Rodents Living in the Home-Cage.</a>

Kahnau et al., *BMC Biology* (2023). <a href="https://link.springer.com/article/10.1186/s12915-023-01751-7" target="_blank">A systematic review of the development and application of home cage monitoring in laboratory mice and rats.</a>

### Social and affective behavior

Ebbesen and Froemke, *Current Opinion in Neurobiology* (2021). <a href="https://www.sciencedirect.com/science/article/pii/S0959438821000106" rel="noopener noreferrer" target="_blank">Body language signals for rodent social communication.</a>

Kuo et al., *Frontiers in Behavioral Neuroscience* (2022). <a href="https://www.frontiersin.org/articles/10.3389/fnbeh.2022.1044492/full" target="_blank">Using deep learning to study emotional behavior in rodent models.</a>

Bordes et al., *Neuroscience & Biobehavioral Reviews* (2023). <a href="https://www.sciencedirect.com/science/article/pii/S0149763423002129" rel="noopener noreferrer"  target="_blank">Advancing social behavioral neuroscience by integrating ethology and comparative psychology methods through machine learning.</a>

### Psychiatry and human behavior

Mobbs et al., *Neuron* (2021). <a href="https://www.sciencedirect.com/science/article/pii/S0896627321003743" rel="noopener noreferrer"  target="_blank">Promises and challenges of human computational ethology.</a>

Shemesh and Chen, *Molecular Psychiatry* (2023). <a href="https://www.nature.com/articles/s41380-022-01913-z" target="_blank">A paradigm shift in translational psychiatry through rodent neuroethology.</a>

Sievers and Thornton, *Social Cognitive and Affective Neuroscience* (2024). <a href="https://academic.oup.com/scan/article/19/1/nsae014/7604387" target="_blank">Deep social neuroscience: the promise and peril of using artificial neural networks to study the social brain.</a>