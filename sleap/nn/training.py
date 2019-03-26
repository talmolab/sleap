import numpy as np
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from sleap.nn.augmentation import Augmenter

from keras.layers import Input, Conv2D, BatchNormalization, Add, MaxPool2D, UpSampling2D, Concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, LambdaCallback

from sleap.nn.architectures.common import conv
from sleap.nn.architectures.hourglass import hourglass, stacked_hourglass
from sleap.nn.architectures.unet import unet
from sleap.nn.architectures.leap import leap_cnn



def train(imgs, confmaps, test_size=0.1, arch="unet", batch_norm=True, num_stacks=1, num_filters=64, batch_size=4, num_epochs=100, steps_per_epoch=200):

    imgs_train, imgs_val, confmaps_train, confmaps_val = train_test_split(imgs, confmaps, test_size=test_size)

    num_train, img_height, img_width, img_channels = imgs_train.shape
    confmaps_channels = confmaps_train.shape[-1]
    print(f"Training set: {imgs_train.shape} -> {confmaps_train.shape}")
    print(f"Validation set: {imgs_val.shape} -> {confmaps_val.shape}")

    # Input layer
    img_input = Input((img_height, img_width, img_channels))

    if arch == "unet":
        x_outs = unet(img_input, confmaps_channels, depth=3, convs_per_depth=2, num_filters=num_filters, interp="bilinear")
    elif arch == "stacked_unet":
        x = img_input
        x_outs = []
        for i in range(num_stacks):
            if i > 0:
                x = Concatenate()([img_input, x])
            x = unet(x, confmaps_channels, depth=3, convs_per_depth=2, num_filters=num_filters, interp="bilinear")
            x_outs.append(x)

    else:
        # Initial downsampling
        x = conv(num_filters, kernel_size=(7, 7))(img_input)
        if batch_norm: x = BatchNormalization()(x)
        # x = residual_block(x, num_filters // 2, batch_norm)
        # x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        # x = residual_block(x, num_filters // 2, batch_norm)
        # x = residual_block(x, num_filters, batch_norm)

        # Stacked hourglass modules
        x_outs = stacked_hourglass(x, confmaps_channels, num_hourglass_blocks=num_stacks, num_filters=num_filters, depth=3, batch_norm=batch_norm, interp="bilinear")

    # Create training model
    model = keras.Model(inputs=img_input, outputs=x_outs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001, amsgrad=True),
        loss="mean_squared_error",
    )
    print("Params: {:,}".format(model.count_params()))


    cmap = np.array([
        [0,   114,   189],
        [217,  83,    25],
        [237, 177,    32],
        [126,  47,   142],
        [119, 172,    48],
        [77,  190,   238],
        [162,  20,    47],
        ]).astype("float32") / 255
    imgs_viz = np.stack((imgs_train[0], imgs_val[0]), axis=0)
    plt.ion()
    fig = plt.figure()
    # plt.show(block=False)
    def plot_preds(epoch, logs):
        preds_viz = model.predict(imgs_viz)
        if not isinstance(preds_viz, np.ndarray):
            preds_viz = preds_viz[-1]
        # preds_viz = preds_viz[model.output_names[-1]]

        fig.clf()
        plt.subplot(2,2,1)
        plt.imshow(imgs_viz[0].squeeze(), cmap="gray")
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,2,3)
        plt.imshow(imgs_viz[1].squeeze(), cmap="gray")
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,2,2)
        for i in range(preds_viz.shape[-1]):
            col = cmap[i % len(cmap)]
            I = preds_viz[0][...,i][...,None] * col[None][None]
            I = np.concatenate((I, preds_viz[0][...,i][...,None]), axis=-1)
            I = (I - np.min(I))/np.ptp(I)
            plt.imshow(I)
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,2,4)
        for i in range(preds_viz.shape[-1]):
            col = cmap[i % len(cmap)]
            I = preds_viz[1][...,i][...,None] * col[None][None]
            I = np.concatenate((I, preds_viz[1][...,i][...,None]), axis=-1)
            I = (I - np.min(I))/np.ptp(I)
            plt.imshow(I)
        plt.xticks([]), plt.yticks([])

        fig.suptitle(f"Training: Epoch {epoch}", fontsize=16)

        # plt.tight_layout()

        # print(logs)

        plt.draw()
        if epoch == -1:
            plt.show(block=False)

        plt.pause(2)


        # plt.draw()
        # plt.show()
    plot_preds(-1, None)

    if steps_per_epoch is None:
        steps_per_epoch = len(imgs_train) // batch_size
    train_datagen = Augmenter(imgs_train, confmaps_train, output_names=model.output_names, batch_size=batch_size)

    monitor_loss = "val_loss"
    # monitor_loss = "val_tf_nn_class_acc"
    callbacks = [
        LambdaCallback(on_epoch_end=plot_preds),
        LambdaCallback(on_epoch_end=lambda epoch, logs: train_datagen.shuffle()),
        # LambdaCallback(on_epoch_end=lambda epoch, logs: check_test_acc()),
        ReduceLROnPlateau(monitor=monitor_loss, min_delta=1e-6, patience=5, verbose=1, factor=0.5, mode="auto", cooldown=3, min_lr=1e-10),
        EarlyStopping(monitor=monitor_loss, min_delta=1e-8, patience=30, verbose=1),
        TensorBoard(log_dir=f"D:/tmp/logs/{arch}{time()}", update_freq=150, histogram_freq=0, batch_size=32, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None),
    #     ModelCheckpoint(filepath="models/best_model.h5", monitor="val_loss", save_best_only=True, verbose=1, period=50),
    ]

    # Train!
    
    training = model.fit_generator(
        train_datagen,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        validation_data=(imgs_val, {output_name: confmaps_val for output_name in model.output_names}),
        callbacks=callbacks,
        verbose=2,
    )

    return model


if __name__ == "__main__":
    import os
    from sleap.io.dataset import Labels, load_labels_json_old
    from sleap.nn.datagen import generate_images, generate_confidence_maps

    data_path = "C:/Users/tdp/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    if not os.path.exists(data_path):
        data_path = "D:/OneDrive/code/sandbox/leap_wt_gold_pilot/centered_pair.json"
    
    labels = load_labels_json_old(data_path)
    imgs = generate_images(labels)
    confmaps = generate_confidence_maps(labels, sigma=5)

    # train(imgs, confmaps, test_size=0.1, batch_norm=False, num_filters=64, batch_size=4, num_epochs=100, steps_per_epoch=100)
    train(imgs, confmaps, test_size=0.1, batch_norm=False, num_filters=32, batch_size=4, num_epochs=100, steps_per_epoch=100, arch="stacked_unet", num_stacks=3)
