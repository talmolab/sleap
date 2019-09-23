import logging
logger = logging.getLogger(__name__)

import numpy as np

from time import time, clock
from typing import Dict, List, Union, Optional, Tuple

import tensorflow as tf
import keras

from sleap.skeleton import Skeleton
from sleap.nn.model import ModelOutputType
from sleap.nn.training import TrainingJob

def load_model(
        sleap_models: List[TrainingJob],
        input_size: Optional[tuple],
        output_types: List[ModelOutputType]) -> keras.Model:
    """
    Load keras Model for specified input size and output types.

    Supports centroids, confmaps, and pafs. If output type includes
    confmaps and pafs then we'll combine these into a single model.

    Arguments:
        sleap_models: dict of the TrainingJobs where we can find models.
        input_size: (h, w, c) tuple; if None, don't resize input layer
        output_types: list of ModelOutputTypes
    Returns:
        keras Model
    """

    if ModelOutputType.CENTROIDS in output_types:
        # Load centroid model
        keras_model = load_model_from_job(sleap_models[ModelOutputType.CENTROIDS])

        logger.info(f"Loaded centroid model trained on shape {keras_model.input_shape}")

    else:
        # Load model for confmaps or pafs or both

        models = []

        new_input_layer = tf.keras.layers.Input(input_size) if input_size is not None else None

        for output_type in output_types:

            # Load the model
            job = sleap_models[output_type]
            model = load_model_from_job(job)

            logger.info(f"Loaded {output_type} model trained on shape {model.input_shape}")

            # Get input layer if we didn't create one for a specified size
            if new_input_layer is None:
                new_input_layer = model.input

            # Resize input layer
            model.layers.pop(0)
            model = model(new_input_layer)

            logger.info(f"  Resized input layer to {input_size}")

            # Add to list of models we've just loaded
            models.append(model)

        if len(models) == 1:
            keras_model = tf.keras.Model(new_input_layer, models[0])
        else:
            # Merge multiple models into single model
            keras_model = tf.keras.Model(new_input_layer, models)

            logger.info(f"  Merged {len(models)} into single model")

    # keras_model = convert_to_gpu_model(keras_model)

    return keras_model

def get_model_data(
            sleap_models: Dict[ModelOutputType,TrainingJob],
            output_types: List[ModelOutputType]) -> Dict:

        model_type = output_types[0]
        job = sleap_models[model_type]

        # Model input is scaled by <multiscale> to get output
        model_properties = dict(
            skeleton=job.model.skeletons[0],
            scale=job.trainer.scale,
            multiscale=job.model.output_scale)

        return model_properties

def get_model_skeleton(sleap_models, output_types) -> Skeleton:

        skeleton = get_model_data(sleap_models, output_types)["skeleton"]

        if skeleton is None:
            logger.warning("Predictor has no skeleton.")
            raise ValueError("Predictor has no skeleton.")

        return skeleton

def load_model_from_job(job: TrainingJob) -> keras.Model:
    """Load keras Model from a specific TrainingJob."""

    # Load model from TrainingJob data
    keras_model = keras.models.load_model(job_model_path(job),
        custom_objects={"tf": tf})

    # Rename to prevent layer naming conflict
    name_prefix = f"{job.model.output_type}_"
    keras_model._name = name_prefix + keras_model.name
    for i in range(len(keras_model.layers)):
        keras_model.layers[i]._name = name_prefix + keras_model.layers[i].name

    return keras_model

def job_model_path(job: TrainingJob) -> str:
    import os
    return os.path.join(job.save_dir, job.best_model_filename)

def get_available_gpus():
    """
    Get the list of available GPUs

    Returns:
        List of available GPU device names
    """

    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def convert_to_gpu_model(model: keras.Model) -> keras.Model:
    gpu_list = get_available_gpus()

    if len(gpu_list) == 0:
        logger.warn('No GPU devices, this is going to be really slow, something is wrong, dont do this!!!')
    else:
        logger.info(f'Detected {len(gpu_list)} GPU(s) for inference')

    if len(gpu_list) > 1:
        model = keras.util.multi_gpu_model(model, gpus=len(gpu_list))

    return model