import pytest
import sleap
from sleap.nn.training import DataReaders


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
    cfg.optimization.online_shuffling = False
    cfg.optimization.prefetch = False
    cfg.optimization.batch_size = 1
    cfg.optimization.batches_per_epoch = 2
    cfg.optimization.val_batches_per_epoch = 2
    cfg.optimization.epochs = 1
    cfg.outputs.save_outputs = False
    return cfg


def test_data_readers(min_tracks_2node_labels):
    labels = min_tracks_2node_labels.extract(range(3), copy=True)
    labels_cfg = sleap.nn.config.LabelsConfig()
    labels_cfg.validation_fraction = 0.1

    # All frames with 2 instances and tracks, no filtering
    data_readers = DataReaders.from_config(labels_config=labels_cfg, training=labels, with_track_only=False)
    assert len(data_readers.training_labels_reader) == 2
    assert len(data_readers.validation_labels_reader) == 1
    assert data_readers.test_labels_reader is None

    # Frames with mixed track and no track, no filtering
    labels = min_tracks_2node_labels.extract(range(3), copy=True)
    labels[0][0].track = None
    labels[1][0].track = None
    labels[2][0].track = None
    data_readers = DataReaders.from_config(labels_config=labels_cfg, training=labels, with_track_only=False)
    assert len(data_readers.training_labels_reader) == 2
    assert len(data_readers.validation_labels_reader) == 1
    examples = list(data_readers.training_labels_reader.make_dataset())
    assert all([ex["instances"].shape[0] == 2 for ex in examples])
    assert all([(ex["track_inds"] == -1).numpy().any() for ex in examples])
    examples = list(data_readers.validation_labels_reader.make_dataset())
    assert all([ex["instances"].shape[0] == 2 for ex in examples])
    assert all([(ex["track_inds"] == -1).numpy().any() for ex in examples])

    # Frames with mixed track and no track, with filtering
    labels = min_tracks_2node_labels.extract(range(3), copy=True)
    labels[0][0].track = None
    labels[1][0].track = None
    labels[2][0].track = None
    data_readers = DataReaders.from_config(labels_config=labels_cfg, training=labels, with_track_only=True)
    assert len(data_readers.training_labels_reader) == 2
    assert len(data_readers.validation_labels_reader) == 1
    examples = list(data_readers.training_labels_reader.make_dataset())
    assert all([ex["instances"].shape[0] == 1 for ex in examples])
    assert all([(ex["track_inds"] >= 0).numpy().all() for ex in examples])
    examples = list(data_readers.validation_labels_reader.make_dataset())
    assert all([ex["instances"].shape[0] == 1 for ex in examples])
    assert all([(ex["track_inds"] >= 0).numpy().all() for ex in examples])

    # Frames with track and frames with no track, with filtering
    labels = min_tracks_2node_labels.extract(range(3), copy=True)
    labels[0][0].track = None
    labels[0][1].track = None
    data_readers = DataReaders.from_config(labels_config=labels_cfg, training=labels, with_track_only=True)
    assert len(data_readers.training_labels_reader) == 1
    assert len(data_readers.validation_labels_reader) == 1
    examples = list(data_readers.training_labels_reader.make_dataset())
    assert all([ex["instances"].shape[0] == 2 for ex in examples])
    assert all([(ex["track_inds"] >= 0).numpy().all() for ex in examples])


def test_train_single_instance(min_labels_robot, cfg):
    cfg.model.heads.single_instance = sleap.nn.config.SingleInstanceConfmapsHeadConfig(
        sigma=1.5, output_stride=1, offset_refinement=False
    )
    trainer = sleap.nn.training.SingleInstanceModelTrainer.from_config(
        cfg, training_labels=min_labels_robot
    )
    trainer.setup()
    trainer.train()
    assert trainer.keras_model.output_names[0] == "SingleInstanceConfmapsHead"
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
    assert trainer.keras_model.output_names[0] == "SingleInstanceConfmapsHead"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 320, 560, 2)

    assert trainer.keras_model.output_names[1] == "OffsetRefinementHead"
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
    assert trainer.keras_model.output_names[0] == "CentroidConfmapsHead"
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
    assert trainer.keras_model.output_names[0] == "CentroidConfmapsHead"
    assert trainer.keras_model.output_names[1] == "OffsetRefinementHead"
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
    assert trainer.keras_model.output_names[0] == "CenteredInstanceConfmapsHead"
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

    assert trainer.keras_model.output_names[0] == "CenteredInstanceConfmapsHead"
    assert trainer.keras_model.output_names[1] == "OffsetRefinementHead"
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

    assert trainer.keras_model.output_names[0] == "MultiInstanceConfmapsHead"
    assert trainer.keras_model.output_names[1] == "PartAffinityFieldsHead"
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

    assert trainer.keras_model.output_names[0] == "MultiInstanceConfmapsHead"
    assert trainer.keras_model.output_names[1] == "PartAffinityFieldsHead"
    assert trainer.keras_model.output_names[2] == "OffsetRefinementHead"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 384, 384, 2)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 192, 192, 2)
    assert tuple(trainer.keras_model.outputs[2].shape) == (None, 384, 384, 4)


def test_train_bottomup_multiclass(min_tracks_2node_labels, cfg):
    labels = min_tracks_2node_labels
    cfg.data.preprocessing.input_scaling = 0.5
    cfg.model.heads.multi_class_bottomup = sleap.nn.config.MultiClassBottomUpConfig(
        confmaps=sleap.nn.config.MultiInstanceConfmapsHeadConfig(
            output_stride=2, offset_refinement=False
        ),
        class_maps=sleap.nn.config.ClassMapsHeadConfig(output_stride=2),
    )
    trainer = sleap.nn.training.BottomUpMultiClassModelTrainer.from_config(
        cfg, training_labels=labels
    )
    trainer.setup()
    trainer.train()

    assert trainer.keras_model.output_names[0] == "MultiInstanceConfmapsHead"
    assert trainer.keras_model.output_names[1] == "ClassMapsHead"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 256, 256, 2)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 256, 256, 2)


def test_train_topdown_multiclass(min_tracks_2node_labels, cfg):
    labels = min_tracks_2node_labels
    cfg.data.instance_cropping.center_on_part = "thorax"
    cfg.model.heads.multi_class_topdown = sleap.nn.config.MultiClassTopDownConfig(
        confmaps=sleap.nn.config.CenteredInstanceConfmapsHeadConfig(
            output_stride=1, offset_refinement=False, anchor_part="thorax"
        ),
        class_vectors=sleap.nn.config.ClassVectorsHeadConfig(output_stride=8),
    )
    trainer = sleap.nn.training.TopDownMultiClassModelTrainer.from_config(
        cfg, training_labels=labels
    )
    trainer.setup()
    trainer.train()

    assert trainer.keras_model.output_names[0] == "CenteredInstanceConfmapsHead"
    assert trainer.keras_model.output_names[1] == "ClassVectorsHead"
    assert tuple(trainer.keras_model.outputs[0].shape) == (None, 64, 64, 2)
    assert tuple(trainer.keras_model.outputs[1].shape) == (None, 2)
