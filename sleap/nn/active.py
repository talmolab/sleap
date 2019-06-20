def make_default_training_jobs():
    from sleap.nn.model import Model, ModelOutputType
    from sleap.nn.training import Trainer, TrainingJob
    from sleap.nn.architectures import unet, leap
    
    # Build Models (wrapper for Keras model with some metadata)
    
    models = dict()
    models[ModelOutputType.CONFIDENCE_MAP] = Model(
            output_type=ModelOutputType.CONFIDENCE_MAP,
            backbone=unet.UNet(num_filters=32))
    models[ModelOutputType.PART_AFFINITY_FIELD] = Model(
            output_type=ModelOutputType.PART_AFFINITY_FIELD,
            backbone=leap.LeapCNN(num_filters=64))

    # Build Trainers

    defaults = dict()
    defaults["shared"] = dict(
            instance_crop = True,
            val_size = 0.1,
            augment_rotation=180,
            batch_size=4,
            learning_rate = 1e-4,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=3,
            reduce_lr_min_delta=1e-6,
            reduce_lr_min_lr = 1e-10,
            amsgrad = True,
            shuffle_every_epoch=True,
            save_every_epoch = False,
#             val_batches_per_epoch = 10,
#             upsampling_layers = True,
#             depth = 3,
    )
    defaults[ModelOutputType.CONFIDENCE_MAP] = dict(
            **defaults["shared"],
            num_epochs=100,
            steps_per_epoch=200,
            reduce_lr_patience=5,
            )

    defaults[ModelOutputType.PART_AFFINITY_FIELD] = dict(
            **defaults["shared"],
            num_epochs=75,
            steps_per_epoch = 100,
            reduce_lr_patience=8,
            )

    trainers = dict()
    for type in models.keys():
        trainers[type] = Trainer(**defaults[type])

    # Build TrainingJobs from Models and Trainers

    training_jobs = dict()
    for type in models.keys():
        training_jobs[type] = TrainingJob(models[type], trainers[type])

    return training_jobs

def run_active_learning_pipeline(labels_filename, labels=None):
    # Imports here so we don't load TensorFlow before necessary
    import os
    from sleap.io.dataset import Labels
    from sleap.nn.training import TrainingJob
    from sleap.nn.model import ModelOutputType
    from sleap.nn.inference import Predictor

    labels = labels or Labels.load_json(labels_filename)

    # If the video frames are large, determine factor to downsample for active learning
    scale = 1
    rescale_under = 1024
    h, w = labels.videos[0].height, labels.videos[0].width
    largest_dim = max(h, w)
    while largest_dim/scale > rescale_under:
        scale += 1

    # Prepare our TrainingJobs

    # Load the defaults we use for active learning
    training_jobs = make_default_training_jobs()

    # Set the parameters specific to this run
    for job in training_jobs.values():
        job.labels_filename = labels_filename
        job.trainer.scale = scale

    # Run the TrainingJobs

    save_dir = os.path.dirname(labels_filename)

    for model_type, job in training_jobs.items():
        run_name = f"models_{model_type}"
        # training_jobs[model_type] = os.path.join(save_dir, run_name+".json")
        training_jobs[model_type] = job.trainer.train(job.model, labels, save_dir=save_dir, run_name=run_name)

    for model_type, job in training_jobs.items():
        # load job from json
        training_jobs[model_type] = TrainingJob.load_json(training_jobs[model_type])

    # Create Predictor from the results of training
    predictor = Predictor.from_training_jobs(training_jobs, labels)

    # Run the Predictor for suggested frames
    # We want to predict for suggested frames that don't already have user instances

    new_labeled_frames = []
    user_labeled_frames = labels.user_labeled_frames

    for video in labels.videos:
        frames = labels.get_video_suggestions(video)
        if len(frames):
            # remove frames that already have user instances
            video_user_labeled_frame_idxs = [lf.frame_idx for lf in user_labeled_frames
                                                if lf.video == video]
            print(f"frames:{frames}, video_user_labeled_frame_idxs:{video_user_labeled_frame_idxs}")
            frames = list(set(frames) - set(video_user_labeled_frame_idxs))
            # get predictions for desired frames in this video
            video_lfs = predictor.predict(input_video=video, frames=frames)
            new_labeled_frames.extend(video_lfs)

    return new_labeled_frames

if __name__ == "__main__":
    import sys
    
#     labels_filename = "/Volumes/fileset-mmurthy/nat/shruthi/labels-mac.json"
    labels_filename = sys.argv[1]

    labeled_frames = run_active_learning_pipeline(labels_filename)
    print(labeled_frames)