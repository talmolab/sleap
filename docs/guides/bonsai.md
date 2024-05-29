(bonsai)=

# Using Bonsai with SLEAP

**Exporting a SLEAP trained model**:

:::{note}
Currently Bonsai only supports the top-down and top-down-id models.
:::

In order to import the trained model into Bonsai, we need to export the model using the {code}`sleap-export` cli command to convert the trained model to a Protocol buffer(.pb) format. For example, for the top-down-id model, the command is as follows:

```
sleap-export -m centroid_model_folder -m top_down_id_model_folder exported_model
```

(for further details please refer {ref} `sleap-export` docs for more details.)

This will generate the necessary `.pb` file and other information files for being used in Bonsai in the exported model folder.

**Installing Bonsai and necessary packages**

- Install bonsai by following the instructions provided in their [installation page](https://bonsai-rx.org/docs/articles/installation.html). 

- Add the necessary packages including the `Bonsai.SLEAP` and `Bonsai.SLEAP.Design` packages for SLEAP to run with Bonsai and to access the SLEAP-Bonsai modules. For more information on other dependency installations, please refer to the official [bonsai sleap documentation](https://github.com/bonsai-rx/sleap?tab=readme-ov-file#bonsai---sleap).

**Using Bonsai SLEAP modules**



