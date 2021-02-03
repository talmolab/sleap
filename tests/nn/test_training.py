import pytest
import sleap

sleap.use_cpu_only()


@pytest.fixture
def training_labels(min_labels):
    labels = min_labels
    labels.append(
        sleap.LabeledFrame(
            video=labels.videos[0], frame_idx=1, instances=labels[0].instances
        )
    )
    return labels


@pytest.fixture
def cfg():
    cfg = sleap.nn.config.TrainingJobConfig()
    cfg.data.instance_cropping.center_on_part = "A"
    cfg.model.backbone.unet = sleap.nn.config.UNetConfig(
        max_stride=8, output_stride=1, filters=8, filters_rate=1.0
    )
    cfg.optimization.preload_data = False
    cfg.optimization.batch_size = 1
    cfg.optimization.batches_per_epoch = 2
    cfg.optimization.epochs = 1
    cfg.outputs.save_outputs = False
    return cfg


def test_train_single_instance(min_labels_robot, cfg):
    cfg.model.heads.single_instance = sleap.nn.config.SingleInstanceConfmapsHeadConfig(
        sigma=1.5, output_stride=1, offset_refinement=False
    )
    trainer = sleap.nn.training.SingleInstanceModelTrainer.from_config(
        cfg, training_labels=min_labels_robot
    )
    trainer.setup()
    trainer.train()
    assert trainer.keras_model.output_names[0] == "SingleInstanceConfmapsHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 320, 560, 2)


def test_train_single_instance_with_offset(min_labels_robot, cfg):
    cfg.model.heads.single_instance = sleap.nn.config.SingleInstanceConfmapsHeadConfig(
        sigma=1.5, output_stride=1, offset_refinement=True
    )
    trainer = sleap.nn.training.SingleInstanceModelTrainer.from_config(
        cfg, training_labels=min_labels_robot
    )
    trainer.setup()
    trainer.train()
    assert trainer.keras_model.output_names[0] == "SingleInstanceConfmapsHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 320, 560, 2)

    assert trainer.keras_model.output_names[1] == "OffsetRefinementHead_0"
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 320, 560, 4)


def test_train_centroids(training_labels, cfg):
    cfg.model.heads.centroid = sleap.nn.config.CentroidsHeadConfig(
        sigma=1.5, output_stride=1, offset_refinement=False
    )
    trainer = sleap.nn.training.CentroidConfmapsModelTrainer.from_config(
        cfg, training_labels=training_labels
    )
    trainer.setup()
    trainer.train()
    assert trainer.keras_model.output_names[0] == "CentroidConfmapsHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 384, 384, 1)


def test_train_centroids_with_offset(training_labels, cfg):
    cfg.model.heads.centroid = sleap.nn.config.CentroidsHeadConfig(
        sigma=1.5, output_stride=1, offset_refinement=True
    )
    trainer = sleap.nn.training.CentroidConfmapsModelTrainer.from_config(
        cfg, training_labels=training_labels
    )
    trainer.setup()
    trainer.train()
    assert trainer.keras_model.output_names[0] == "CentroidConfmapsHead_0"
    assert trainer.keras_model.output_names[1] == "OffsetRefinementHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 384, 384, 1)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 384, 384, 2)


def test_train_topdown(training_labels, cfg):
    cfg.model.heads.centered_instance = (
        sleap.nn.config.CenteredInstanceConfmapsHeadConfig(
            sigma=1.5, output_stride=1, offset_refinement=False
        )
    )
    trainer = sleap.nn.training.TopdownConfmapsModelTrainer.from_config(
        cfg, training_labels=training_labels
    )
    trainer.setup()
    trainer.train()
    assert trainer.keras_model.output_names[0] == "CenteredInstanceConfmapsHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 96, 96, 2)


def test_train_topdown_with_offset(training_labels, cfg):
    cfg.model.heads.centered_instance = (
        sleap.nn.config.CenteredInstanceConfmapsHeadConfig(
            sigma=1.5, output_stride=1, offset_refinement=True
        )
    )
    trainer = sleap.nn.training.TopdownConfmapsModelTrainer.from_config(
        cfg, training_labels=training_labels
    )
    trainer.setup()
    trainer.train()

    assert trainer.keras_model.output_names[0] == "CenteredInstanceConfmapsHead_0"
    assert trainer.keras_model.output_names[1] == "OffsetRefinementHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 96, 96, 2)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 96, 96, 4)


def test_train_bottomup(training_labels, cfg):
    cfg.model.heads.multi_instance = sleap.nn.config.MultiInstanceConfig(
        confmaps=sleap.nn.config.MultiInstanceConfmapsHeadConfig(
            output_stride=1, offset_refinement=False
        ),
        pafs=sleap.nn.config.PartAffinityFieldsHeadConfig(output_stride=2),
    )
    trainer = sleap.nn.training.TopdownConfmapsModelTrainer.from_config(
        cfg, training_labels=training_labels
    )
    trainer.setup()
    trainer.train()

    assert trainer.keras_model.output_names[0] == "MultiInstanceConfmapsHead_0"
    assert trainer.keras_model.output_names[1] == "PartAffinityFieldsHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 384, 384, 2)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 192, 192, 2)


def test_train_bottomup_with_offset(training_labels, cfg):
    cfg.model.heads.multi_instance = sleap.nn.config.MultiInstanceConfig(
        confmaps=sleap.nn.config.MultiInstanceConfmapsHeadConfig(
            output_stride=1, offset_refinement=True
        ),
        pafs=sleap.nn.config.PartAffinityFieldsHeadConfig(output_stride=2),
    )
    trainer = sleap.nn.training.TopdownConfmapsModelTrainer.from_config(
        cfg, training_labels=training_labels
    )
    trainer.setup()
    trainer.train()

    assert trainer.keras_model.output_names[0] == "MultiInstanceConfmapsHead_0"
    assert trainer.keras_model.output_names[1] == "PartAffinityFieldsHead_0"
    assert trainer.keras_model.output_names[2] == "OffsetRefinementHead_0"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 384, 384, 2)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 192, 192, 2)
    assert tuple(trainer.keras_model.outputs[2].shape) == (None, 384, 384, 4)
