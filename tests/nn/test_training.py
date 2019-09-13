import os

import pytest

from sleap.nn.model import Model, ModelOutputType
from sleap.nn.architectures import *
from sleap.nn.architectures.leap import leap_cnn
from sleap.nn.training import Trainer, TrainingJob

def test_model_fail_non_available_backbone(multi_skel_vid_labels):
    with pytest.raises(ValueError):
        Model(output_type=ModelOutputType.CONFIDENCE_MAP, backbone=object(),
              skeletons=multi_skel_vid_labels.skeletons)


@pytest.mark.parametrize("backbone", available_archs)
def test_training_job_json(tmpdir, multi_skel_vid_labels, backbone):
    run_name = 'training'

    model = Model(output_type=ModelOutputType.CONFIDENCE_MAP, backbone=backbone(),
              skeletons=multi_skel_vid_labels.skeletons)

    train_run = TrainingJob(model=model, trainer=Trainer(),
                            save_dir=os.path.join(tmpdir), run_name=run_name)

    # Create and serialize training info
    json_path = os.path.join(tmpdir, f"{run_name}.json")
    TrainingJob.save_json(train_run, json_path)

    # Load the JSON back in
    loaded_run = TrainingJob.load_json(json_path)

    # Make sure the skeletons match (even though not eq)
    for sk1, sk2 in zip(loaded_run.model.skeletons, train_run.model.skeletons):
        assert sk1.matches(sk2)

    # Now remove the skeletons since we want to check eq on everything else
    loaded_run.model.skeletons = []
    train_run.model.skeletons = []

    assert loaded_run == train_run

