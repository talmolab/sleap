(bonsai)=

# Using Bonsai with SLEAP

Bonsai is a visual language for reactive programming and currently supports SLEAP models.

:::{note}
Currently Bonsai supports only single instance, top-down and top-down-id SLEAP models.
:::

### Exporting a SLEAP trained model:

Before we can import a trained model into Bonsai, we need to use the {code}`sleap-export` command to convert the model to a format supported by Bonsai. For example, for the top-down-id model, the command is as follows:

```
sleap-export -m centroid_model_folder_path -m top_down_id_model_folder_path exported_model
```

Please refer to the {ref}`sleap-export` docs for more details on using the command.

This will generate the necessary `.pb` file and other information files required by Bonsai. In this example, these files were saved to the specified `exported_model` folder.

### Installing Bonsai and necessary packages:

1. Install Bonsai. See the [Bonsai installation instructions](https://bonsai-rx.org/docs/articles/installation.html). 

2. Download and add the necessary packages for Bonsai to run with SLEAP. See the official [Bonsai SLEAP documentation](https://github.com/bonsai-rx/sleap?tab=readme-ov-file#bonsai---sleap) for more information.

### Using Bonsai SLEAP modules:

Once you have Bonsai installed with the required packages, you should be able to open the Bonsai application.  

- The workflow must have an source module `FileCapture` which can be found in the toolbox search in the workflow editor. Provide the path to the video that was used to train the SLEAP model in the `FileName` field of the module.

#### Top-down model:
The top-down model requires both the `PredictCentroid` and the `PredictPoses` modules.

The `PredictCentroid` module will predict the centroids of detections. There are two fields inside the `PredictCentroid` module: the `ModelFileName` field and the `TrainingConfig` field. The `TrainingConfig` field expects the path to the training config JSON file for the centroid model. The `ModelFileName` field expects the path to the `exported_model` folder (which contains the exported `.pb` file).

The `PredictPoses` module will predict the instances of detections. Similar to the `PredictCentroid` module, there are two fields inside the `PredictPoses` module: the `ModelFileName` field and the `TrainingConfig` field. The `TrainingConfig` field expects the path to the training config JSON file for the centered instance model. The `ModelFileName` field expects the path to the `exported_model` folder (which contains the exported `.pb` file).

#### Top-Down-ID model:
The `PredictPoseIdentities` module will predict the instances with identities. This module has two fields: the `ModelFileName` field and the `TrainingConfig` field. The `TrainingConfig` field expects the path to the training config JSON file for the top-down-id model. The `ModelFileName` field expects the path to the `exported_model` folder (which contains the exported `.pb` file).

#### Single instance model:
The `PredictSinglePose` module will predict the poses for single instance models. This module also has two fields: the `ModelFileName` field and the `TrainingConfig` field. The `TrainingConfig` field expects the path to the training config JSON file for the single instance model. The `ModelFileName` field expects the path to the `exported_model` folder (which contains the exported `.pb` file).


The workflow in Bonsai will look something like the following:

![Bonsai.SLEAP workflow](../_static/bonsai-workflow.jpg)

- Once you have the basic workflow for Bonsai.SLEAP created and running successfully, you can add more modules to analyze and visualize the results in Bonsai.

For more documentation on various modules and workflows, please refer to the [official Bonsai docs](https://bonsai-rx.org/docs/articles/editor.html).