import os
import numpy as np
from numpy.random import randint
import random
import tensorflow as tf
from readfiles import *
from models import *
from lossFunctions import *
from math import log
import sys
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import layers

def getIndices(idx1, idx2):
    if idx1 == 0:
        return [*range(1, idx2, 1)]
    elif idx1 + 1 == idx2:
        return [*range(0, idx1, 1)]
    else:
        return [*range(0, idx1), *range(idx1 + 1, idx2)]


def getData(data, indices):
    return [data[i] for i in indices]


def trainModel(train_dataset, validation_dataset, itrs):
    model = resnet3d()
    initial_learning_rate = 0.0001
    model.compile(
        loss=ssim_loss,
        optimizer=keras.optimizers.Nadam(initial_learning_rate),
        metrics=ssim_loss,
    )
    print("Model Compiled")

    modelName = "resnet3d_" + itrs + ".h5"
    print(modelName)

    # Save best model
    checkpoint_cb = keras.callbacks.ModelCheckpoint(modelName, save_best_only=True)

    # Early stopping
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )

    # Train the model, doing validation at the end of each epoch
    epochs = 1000
    print("start fit")
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )


def define_gan(g_model, d_model, image_shape=(None, None, None, 2)):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = keras.Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = keras.models.Model(in_src, [dis_out, gen_out])
    # compile model
    lr = 0.01
    opt = keras.optimizers.Nadam(lr)
    model.compile(
        loss=["binary_crossentropy", ssim_loss], optimizer=opt, loss_weights=[0.5, 50]
    )
    return model


def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1)) * round(
        random.uniform(0.9, 1.0), 1
    )
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = np.ones((len(X), patch_shape, patch_shape, 1)) * round(
        random.uniform(0.0, 0.1), 1
    )
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, itrs, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = X_realA[:, :, :, 1]

    pyplot.ioff()

    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(X_fakeB[i, :, :, 60])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(X_realB[i, :, :, 60])
    # save plot to file
    filename1 = "./gansbrtfig_" + itrs + "/plot_%06d.png" % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = "./gansbrtmodel_" + itrs + "/model_%06d.h5" % (step + 1)
    g_model.save(filename2)
    tf.print(">Saved: %s and %s" % (filename1, filename2), output_stream=sys.stdout)


def train(d_model, g_model, gan_model, dataset, itrs, n_epochs=1000, n_batch=1):
    # unpack dataset
    trainA, trainB = dataset
    n_patch = trainA.shape[1] // (2 ** 4)
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    tf.print("Start Training", output_stream=sys.stdout)
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        if i % 2 == 0:
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        tf.print(
            ">%d, d1[%.3f] d2[%.3f] g[%.3f]" % (i + 1, d_loss1, d_loss2, g_loss),
            output_stream=sys.stdout,
        )
        # summarize model performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset, itrs)


def finetunegan(dataset, modelfile):
    d_model = define_discriminator()
    g_model = keras.models.load_model(modelfile)
    gan_model = define_gan(g_model, d_model)
    train(d_model, g_model, gan_model, dataset, 1)


def traingan(dataset, itrs):
    # define the models
    x, y = dataset
    d_model = define_discriminator()
    g_model = resnet3d()
    # define the composite model
    gan_model = define_gan(g_model, d_model)
    # train model
    train(d_model, g_model, gan_model, dataset, itrs)


def main():
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    itrs = "1"
    dose_scan_paths = [
        os.path.join(os.getcwd(), "mldata/sbrt_dose", x)
        for x in sorted(os.listdir("mldata/sbrt_dose"))
        if not x.startswith(".")
    ]
    prejac_scan_paths = [
        os.path.join(os.getcwd(), "mldata/sbrt_prejac", x)
        for x in sorted(os.listdir("mldata/sbrt_prejac"))
        if not x.startswith(".")
    ]

    postjac_scan_paths = [
        os.path.join(os.getcwd(), "mldata/sbrt_postjac", x)
        for x in sorted(os.listdir("mldata/sbrt_postjac"))
        if not x.startswith(".")
    ]

    w = 128
    h = 128
    d = 128

    # Read and process the scans.
    # Each scan is resized across height, width, and depth and rescaled.
    dose_scans = np.array([process_scan(path, 3, w, h, d) for path in dose_scan_paths])
    prejac_scans = np.array(
        [process_scan(path, 1, w, h, d) for path in prejac_scan_paths]
    )
    postjac_scans = np.array(
        [process_scan(path, 1, w, h, d) for path in postjac_scan_paths]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_scans = postjac_scans / prejac_scans
        ratio_scans = normalize_ratio(ratio_scans)
    ratio_scans = tf.expand_dims(ratio_scans, axis=4)

    usegan = True
    finetune = True
    modelfile = ""

    if usegan:
        x_train = np.stack((dose_scans, prejac_scans), axis=4)
        y_train = np.stack(ratio_scans, axis=0)
        tf.print(
            "Number of samples in train are %d." % (x_train.shape[0]),
            output_stream=sys.stdout,
        )
        if finetune:
            finetunegan([x_train, y_train], modelfile)
        else:
            traingan([x_train, y_train], itrs)
    else:

        indices = np.random.RandomState(seed=42).permutation(prejac_scans.shape[0])
        eighty = math.floor(len(prejac_scans) * 0.8)
        training_idx, test_idx = indices[:eighty], indices[eighty:]
        x_train = np.stack(
            (
                getData(dose_scans, training_idx),
                getData(prejac_scans, training_idx),
            ),
            axis=4,
        )
        y_train = np.stack(getData(ratio_scans, training_idx), axis=0)
        x_val = np.stack(
            (
                getData(dose_scans, test_idx),
                getData(prejac_scans, test_idx),
            ),
            axis=4,
        )
        y_val = np.stack(getData(ratio_scans, test_idx), axis=0)
        print(
            "Number of samples in train and validation are %d and %d."
            % (x_train.shape[0], x_val.shape[0])
        )

        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        batch_size = 2
        # Augment the on the fly during training.
        train_dataset = (
            train_loader.shuffle(len(x_train))
            .map(train_preprocessing)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(x_val))
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        trainModel(train_dataset, validation_dataset, itrs)


if __name__ == "__main__":
    main()
