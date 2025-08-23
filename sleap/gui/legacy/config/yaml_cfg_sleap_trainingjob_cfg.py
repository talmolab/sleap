from omegaconf import OmegaConf
from sleap.gui.legacy.config import (
    TrainingJobConfig,
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    OutputsConfig,
    LabelsConfig,
    PreprocessingConfig,
    InstanceCroppingConfig,
    BackboneConfig,
    HeadsConfig,
    UNetConfig,
    SingleInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
    MultiClassBottomUpConfig,
    MultiClassTopDownConfig,
    ClassMapsHeadConfig,
    ClassVectorsHeadConfig,
    AugmentationConfig,
    LearningRateScheduleConfig,
    HardKeypointMiningConfig,
    EarlyStoppingConfig,
)

from sleap.gui.legacy.config.outputs import CheckpointingConfig, ZMQConfig


def mapper(config: OmegaConf):
    from sleap_io.io.skeleton import SkeletonEncoder, SkeletonYAMLDecoder

    data_cfg = config.data_config
    backbone_cfg = config.model_config.backbone_config
    head_cfgs = config.model_config.head_configs
    trainer_cfg = config.trainer_config
    crop_size = OmegaConf.select(data_cfg, "preprocessing.crop_hw", default=None)

    skeletons = None
    if data_cfg.skeletons is not None and len(data_cfg.skeletons) > 0:
        skeletons = []
        for skel_cfg in data_cfg.skeletons:
            skeleton = SkeletonYAMLDecoder().decode(dict(skel_cfg))
            skeleton.name = skel_cfg.name
            skeletons.append(skeleton)

    data = DataConfig(
        labels=LabelsConfig(
            training_labels=(
                data_cfg.train_labels_path[0]
                if data_cfg.train_labels_path is not None
                and len(data_cfg.train_labels_path)
                else None
            ),
            validation_labels=(
                data_cfg.val_labels_path[0]
                if data_cfg.val_labels_path is not None
                and len(data_cfg.val_labels_path)
                else None
            ),
            validation_fraction=data_cfg.validation_fraction,
            test_labels=data_cfg.test_file_path,
            skeletons=(
                SkeletonEncoder().encode(skeletons) if skeletons is not None else None
            ),
        ),
        preprocessing=PreprocessingConfig(
            ensure_rgb=data_cfg.preprocessing.ensure_rgb,
            ensure_grayscale=data_cfg.preprocessing.ensure_grayscale,
            input_scaling=data_cfg.preprocessing.scale,
            resize_and_pad_to_target=True,
            target_height=data_cfg.preprocessing.max_height,
            target_width=data_cfg.preprocessing.max_width,
        ),
        instance_cropping=InstanceCroppingConfig(
            crop_size=crop_size[0] if crop_size is not None else None
        ),
    )

    # head type
    for head in config.model_config.head_configs:
        if config.model_config.head_configs[f"{head}"] is not None:
            head_type = head
            break

    model = ModelConfig(
        backbone=BackboneConfig(
            unet=UNetConfig(
                stem_stride=backbone_cfg.unet.stem_stride,
                max_stride=backbone_cfg.unet.max_stride,
                output_stride=backbone_cfg.unet.output_stride,
                filters=backbone_cfg.unet.filters,
                filters_rate=backbone_cfg.unet.filters_rate,
                middle_block=backbone_cfg.unet.middle_block,
                up_interpolate=backbone_cfg.unet.up_interpolate,
                stacks=backbone_cfg.unet.stacks,
            )
        ),
        heads=HeadsConfig(
            single_instance=SingleInstanceConfmapsHeadConfig(
                part_names=[x for x in head_cfgs.single_instance.confmaps.part_names],
                sigma=head_cfgs.single_instance.confmaps.sigma,
                output_stride=head_cfgs.single_instance.confmaps.output_stride,
            )
            if head_type == "single_instance"
            else None,
            centroid=CentroidsHeadConfig(
                anchor_part=head_cfgs.centroid.confmaps.anchor_part,
                sigma=head_cfgs.centroid.confmaps.sigma,
                output_stride=head_cfgs.centroid.confmaps.output_stride,
            )
            if head_type == "centroid"
            else None,
            centered_instance=CenteredInstanceConfmapsHeadConfig(
                part_names=[x for x in head_cfgs.centered_instance.confmaps.part_names],
                anchor_part=head_cfgs.centered_instance.confmaps.anchor_part,
                sigma=head_cfgs.centered_instance.confmaps.sigma,
                output_stride=head_cfgs.centered_instance.confmaps.output_stride,
                loss_weight=head_cfgs.centered_instance.confmaps.loss_weight,
            )
            if head_type == "centered_instance"
            else None,
            multi_instance=MultiInstanceConfig(
                confmaps=MultiInstanceConfmapsHeadConfig(
                    part_names=[x for x in head_cfgs.bottomup.confmaps.part_names],
                    sigma=head_cfgs.bottomup.confmaps.sigma,
                    output_stride=head_cfgs.bottomup.confmaps.output_stride,
                    loss_weight=head_cfgs.bottomup.confmaps.loss_weight,
                ),
                pafs=PartAffinityFieldsHeadConfig(
                    edges=[edge for edge in head_cfgs.bottomup.pafs.edges],
                    sigma=head_cfgs.bottomup.pafs.sigma,
                    output_stride=head_cfgs.bottomup.pafs.output_stride,
                    loss_weight=head_cfgs.bottomup.pafs.loss_weight,
                ),
            )
            if head_type == "bottomup"
            else None,
            multi_class_bottomup=MultiClassBottomUpConfig(
                confmaps=MultiInstanceConfmapsHeadConfig(
                    part_names=[
                        x for x in head_cfgs.multi_class_bottomup.confmaps.part_names
                    ],
                    sigma=head_cfgs.multi_class_bottomup.confmaps.sigma,
                    output_stride=head_cfgs.multi_class_bottomup.confmaps.output_stride,
                    loss_weight=head_cfgs.multi_class_bottomup.confmaps.loss_weight,
                ),
                class_maps=ClassMapsHeadConfig(
                    classes=[
                        x for x in head_cfgs.multi_class_bottomup.class_maps.classes
                    ],
                    sigma=head_cfgs.multi_class_bottomup.class_maps.sigma,
                    output_stride=head_cfgs.multi_class_bottomup.class_maps.output_stride,
                    loss_weight=head_cfgs.multi_class_bottomup.class_maps.loss_weight,
                ),
            )
            if head_type == "multi_class_bottomup"
            else None,
            multi_class_topdown=MultiClassTopDownConfig(
                confmaps=CenteredInstanceConfmapsHeadConfig(
                    part_names=[
                        x for x in head_cfgs.multi_class_topdown.confmaps.part_names
                    ],
                    anchor_part=head_cfgs.multi_class_topdown.confmaps.anchor_part,
                    sigma=head_cfgs.multi_class_topdown.confmaps.sigma,
                    output_stride=head_cfgs.multi_class_topdown.confmaps.output_stride,
                    loss_weight=head_cfgs.multi_class_topdown.confmaps.loss_weight,
                ),
                class_vectors=ClassVectorsHeadConfig(
                    classes=[
                        x for x in head_cfgs.multi_class_topdown.class_vectors.classes
                    ],
                    num_fc_layers=head_cfgs.multi_class_topdown.class_vectors.num_fc_layers,
                    num_fc_units=head_cfgs.multi_class_topdown.class_vectors.num_fc_units,
                    global_pool=head_cfgs.multi_class_topdown.class_vectors.global_pool,
                    output_stride=head_cfgs.multi_class_topdown.class_vectors.output_stride,
                    loss_weight=head_cfgs.multi_class_topdown.class_vectors.loss_weight,
                ),
            )
            if head_type == "multi_class_topdown"
            else None,
        ),
    )
    aug_cfg = data_cfg.augmentation_config
    ac = AugmentationConfig(
        rotate=True if aug_cfg.geometric.affine_p > 0 else False,
        rotation_min_angle=-aug_cfg.geometric.rotation_min,
        rotation_max_angle=aug_cfg.geometric.rotation_max,
        scale=True if aug_cfg.geometric.affine_p > 0 else False,
        scale_min=aug_cfg.geometric.scale_min,
        scale_max=aug_cfg.geometric.scale_max,
        uniform_noise=True if aug_cfg.intensity.uniform_noise_p > 0 else False,
        uniform_noise_min_val=aug_cfg.intensity.uniform_noise_min,
        uniform_noise_max_val=aug_cfg.intensity.uniform_noise_max,
        gaussian_noise=True if aug_cfg.intensity.gaussian_noise_p > 0 else False,
        gaussian_noise_mean=aug_cfg.intensity.gaussian_noise_mean,
        gaussian_noise_stddev=aug_cfg.intensity.gaussian_noise_std,
        contrast=True if aug_cfg.intensity.contrast_p > 0 else False,
        contrast_min_gamma=aug_cfg.intensity.contrast_min,
        contrast_max_gamma=aug_cfg.intensity.contrast_max,
        brightness=True if aug_cfg.intensity.brightness_p > 0 else False,
        brightness_min_val=aug_cfg.intensity.brightness_min,
        brightness_max_val=aug_cfg.intensity.brightness_max,
    )
    augmentaion_config = (
        AugmentationConfig() if not data_cfg.use_augmentations_train else ac
    )
    optimization = OptimizationConfig(
        augmentation_config=augmentaion_config,
        batch_size=trainer_cfg.train_data_loader.batch_size,
        epochs=trainer_cfg.max_epochs,
        optimizer=trainer_cfg.optimizer_name,
        min_batches_per_epoch=trainer_cfg.min_train_steps_per_epoch,
        batches_per_epoch=trainer_cfg.train_steps_per_epoch,
        initial_learning_rate=trainer_cfg.optimizer.lr,
        learning_rate_schedule=LearningRateScheduleConfig(
            reduce_on_plateau=True
            if trainer_cfg.lr_scheduler.reduce_lr_on_plateau is not None
            else False,
            reduction_factor=trainer_cfg.lr_scheduler.reduce_lr_on_plateau.factor,
            plateau_min_delta=trainer_cfg.lr_scheduler.reduce_lr_on_plateau.factor,
            plateau_patience=trainer_cfg.lr_scheduler.reduce_lr_on_plateau.patience,
            plateau_cooldown=trainer_cfg.lr_scheduler.reduce_lr_on_plateau.cooldown,
            min_learning_rate=trainer_cfg.lr_scheduler.reduce_lr_on_plateau.min_lr,
        ),
        hard_keypoint_mining=HardKeypointMiningConfig(
            online_mining=trainer_cfg.online_hard_keypoint_mining.online_mining,
            hard_to_easy_ratio=trainer_cfg.online_hard_keypoint_mining.hard_to_easy_ratio,
            min_hard_keypoints=trainer_cfg.online_hard_keypoint_mining.min_hard_keypoints,
            max_hard_keypoints=trainer_cfg.online_hard_keypoint_mining.max_hard_keypoints,
            loss_scale=trainer_cfg.online_hard_keypoint_mining.loss_scale,
        ),
        early_stopping=EarlyStoppingConfig(
            stop_training_on_plateau=trainer_cfg.early_stopping.stop_training_on_plateau,
            plateau_min_delta=trainer_cfg.early_stopping.min_delta,
            plateau_patience=trainer_cfg.early_stopping.patience,
        ),
    )
    outputs = OutputsConfig(
        run_name=trainer_cfg.save_ckpt_path.split("/")[-1],
        save_visualizations=trainer_cfg.visualize_preds_during_training,
        keep_viz_images=trainer_cfg.keep_viz,
        checkpointing=CheckpointingConfig(
            best_model=trainer_cfg.save_ckpt,
            latest_model=trainer_cfg.model_ckpt.save_last,
        ),
        zmq=ZMQConfig(
            subscribe_to_controller=True
            if trainer_cfg.zmq.controller_address is not None
            else False,
            controller_address=trainer_cfg.zmq.controller_address,
            publish_updates=True
            if trainer_cfg.zmq.publish_address is not None
            else False,
            publish_address=trainer_cfg.zmq.publish_address,
            controller_polling_timeout=trainer_cfg.zmq.controller_polling_timeout,
        ),
    )
    filename = trainer_cfg.save_ckpt_path + "/training_config.yaml"

    return TrainingJobConfig(
        data=data,
        model=model,
        optimization=optimization,
        outputs=outputs,
        filename=filename,
    )
