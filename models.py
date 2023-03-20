import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


def upsample2(x, upsamplescale, channels):
    deconv = layers.Conv3DTranspose(
        filters=1,
        kernel_size=upsamplescale,
        strides=upsamplescale,
        padding="same",
        name="UpsampleDeconv" + str(upsamplescale),
    )(x)
    divide_by = upsamplescale ** 3
    return layers.Conv3D(
        filters=1,
        kernel_size=upsamplescale,
        padding="same",
        name="UpSampleSmooth" + str(upsamplescale),
    )(deconv)


def upsample(x, upsamplescale, channel_count):
    x_shape = tf.shape(x)
    output_shape = tf.stack(
        [
            x_shape[0],
            x_shape[1] * upsamplescale,
            x_shape[2] * upsamplescale,
            x_shape[3] * upsamplescale,
            channel_count,
        ]
    )
    constant5d = tf.cast(
        tf.constant(
            np.ones(
                [
                    upsamplescale,
                    upsamplescale,
                    upsamplescale,
                    channel_count,
                    channel_count,
                ]
            )
        ),
        tf.float32,
    )
    deconv = tf.nn.conv3d_transpose(
        x,
        filters=constant5d,
        output_shape=output_shape,
        strides=[1, upsamplescale, upsamplescale, upsamplescale, 1],
        padding="SAME",
        name="UpsampleDeconv",
    )
    divide_by = upsamplescale ** 3
    val = np.array(1, np.float32) / np.array(divide_by, np.float32)
    smooth5d = tf.cast(
        tf.constant(
            val
            * np.ones(
                [
                    upsamplescale,
                    upsamplescale,
                    upsamplescale,
                    channel_count,
                    channel_count,
                ]
            ),
            name="Upsample" + str(upsamplescale),
        ),
        tf.float32,
    )
    return tf.nn.conv3d(
        deconv,
        filters=smooth5d,
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
        name="UpsampleSmooth" + str(upsamplescale),
    )


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            activation=layers.LeakyReLU(alpha=0.2),
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    return result


def define_discriminator(image_shape=(None, None, None, 2)):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # source image input
    in_src_image = keras.Input(shape=image_shape)
    # target image input
    in_target_image = keras.Input(
        shape=(image_shape[0], image_shape[1], image_shape[2], 1)
    )
    # concatenate images channel-wise
    merged = layers.Concatenate()([in_src_image, in_target_image])
    # C64
    d = layers.Conv3D(
        64, kernel_size=4, strides=2, padding="same", kernel_initializer=init
    )(merged)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # C128
    d = layers.Conv3D(
        128, kernel_size=4, strides=2, padding="same", kernel_initializer=init
    )(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # C256
    d = layers.Conv3D(
        256, kernel_size=4, strides=2, padding="same", kernel_initializer=init
    )(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # C512
    d = layers.Conv3D(
        512, kernel_size=4, strides=2, padding="same", kernel_initializer=init
    )(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = layers.Conv3D(512, kernel_size=4, padding="same", kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # patch output
    d = layers.Conv3D(1, kernel_size=4, padding="same", kernel_initializer=init)(d)
    patch_out = layers.Activation("sigmoid")(d)
    # define model
    model = keras.models.Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = keras.optimizers.Nadam(learning_rate=0.00001)
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # add downsampling layer
    g = layers.Conv3D(
        filters=n_filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=init,
    )(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = layers.BatchNormalization()(g, training=True)
    # leaky relu activation
    g = layers.LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # add upsampling layer
    g = layers.Conv3DTranspose(
        filters=n_filters,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=init,
    )(layer_in)
    # add batch normalization
    g = layers.BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = layers.Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = layers.Concatenate()([g, skip_in])
    # relu activation
    g = layers.Activation("relu")(g)
    return g


def define_generator(image_shape=(None, None, None, 2)):
    """Build a 3D convolutional neural network model based on Ye et al."""
    channels = 2
    layernum = 5
    inputs = keras.Input(image_shape)
    convLayers = []
    concatLayers = []
    upsampleLayers = []
    for layerval in range(layernum):
        if layerval == 0:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(inputs)
        else:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(
            filters=2 ** (5 + layerval),
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Conv2",
        )(x)
        convLayers.append(layers.BatchNormalization()(x))
        if layerval < layernum - 1:
            x = layers.MaxPool3D(pool_size=2, strides=2)(x)

    for layerval in range(layernum - 1, 0, -1):
        if layerval == layernum - 1:
            concat = convLayers[layerval]
        upsampled = layers.UpSampling3D(size=2)(concat)
        concat = layers.concatenate([upsampled, convLayers[layerval - 1]], axis=4)
        concat = layers.Conv3D(
            filters=2 ** (4 + layerval),
            kernel_size=3,
            activation=layers.ReLU,
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Concat1",
        )(concat)
        concat = layers.BatchNormalization()(concat)
        concat = layers.Conv3D(
            filters=2 ** (4 + layerval),
            kernel_size=3,
            activation=layers.ReLU,
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Concat2",
        )(concat)
        concat = layers.BatchNormalization()(concat)

    prev_layer = layers.Conv3D(
        filters=1, kernel_size=3, activation="sigmoid", padding="same"
    )(concat)
    model = keras.Model(inputs, prev_layer, name="unet")
    return model

    prev_layer = layers.Conv3D(
        filters=1,
        kernel_size=3,
        padding="same",
        name="output1",
        activation=keras.activations.sigmoid,
    )(concat)
    model = keras.Model(inputs, prev_layer, name="3dcnn")
    return model


def afterconv(x):
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2, 3])(x)
    x = layers.Dropout(0.1)(x)
    return x


def resconv(x, filternum):
    conv1 = afterconv(
        layers.Conv3D(
            filters=filternum,
            kernel_size=5,
            kernel_initializer="he_normal",
            padding="same",
        )(x)
    )
    conv2 = afterconv(
        layers.Conv3D(
            filters=filternum,
            kernel_size=5,
            kernel_initializer="he_normal",
            padding="same",
        )(conv1)
    )
    conv3 = afterconv(
        layers.Conv3D(
            filters=filternum,
            kernel_size=5,
            kernel_initializer="he_normal",
            padding="same",
        )(conv2)
    )
    x = afterconv(
        layers.Conv3D(
            filters=filternum,
            kernel_size=1,
            kernel_initializer="he_normal",
            padding="same",
        )(x)
    )
    return layers.Add()([x, conv3])


def skipconv(skip, x, filternum):
    filternum = filternum // 2
    up = layers.UpSampling3D(2)(x)
    theta_x = layers.Conv3D(
        filters=filternum, kernel_size=1, kernel_initializer="he_normal", padding="same"
    )(skip)
    phi_g = layers.Conv3D(
        filters=filternum, kernel_size=1, kernel_initializer="he_normal", padding="same"
    )(up)
    f = layers.ReLU()(layers.Add()([theta_x, phi_g]))
    psi_f = layers.Conv3D(
        filters=1, kernel_size=1, kernel_initializer="he_normal", padding="same"
    )(f)
    rate = layers.Activation("sigmoid")(psi_f)
    att_x = layers.Multiply()([skip, rate])
    return att_x


def rennet(image_shape=(None, None, None, 2)):
    layernum = 3
    maxfilter = 2 ** (layernum + 4)
    inputs = keras.Input(image_shape)
    convLayers = []
    x = layers.Conv3D(
        filters=8, kernel_size=5, kernel_initializer="he_normal", padding="same"
    )(inputs)
    for layerval in range(layernum):
        x = layers.Conv3D(
            filters=2 ** (layerval + 4),
            kernel_size=5,
            strides=2,
            kernel_initializer="he_normal",
            padding="same",
        )(x)
        x = afterconv(x)
        x = resconv(x, 2 ** (layerval + 4))
        if layerval < 2:
            convLayers.append(x)
    x = resconv(x, maxfilter)
    x1 = resconv(x, maxfilter // 2)
    for layerval in range(layernum):
        x = layers.UpSampling3D(2)(x1)
        x = layers.Conv3D(
            filters=2 ** (5 - layerval),
            kernel_size=5,
            kernel_initializer="he_normal",
            padding="same",
        )(x)
        x = afterconv(x)
        if layerval < 2:
            x = resconv(x, 2 ** (5 - layerval))
            skip = convLayers.pop()
            skip = skipconv(skip, x1, 2 ** (5 - layerval))
            x1 = layers.Concatenate(axis=-1)([x, skip])
            x1 = afterconv(
                layers.Conv3D(
                    filters=2 ** (5 - layerval),
                    kernel_size=5,
                    kernel_initializer="he_normal",
                    padding="same",
                )(x1)
            )
        else:
            x = resconv(x, 1)
            x = keras.activations.sigmoid(x)
    model = keras.models.Model(inputs, x)
    return model


# define the standalone generator model
def define_generator2(image_shape=(None, None, None, 2)):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # image input
    inputs = keras.Input(shape=image_shape)
    x = inputs
    # Dictionary for long connections
    long_connection_store = {}
    Conv = layers.Conv3D
    MaxPooling = layers.MaxPooling3D
    UpSampling = layers.UpSampling3D
    depth = 5
    filter_root = 64
    norm = True
    # Down sampling
    for i in range(depth):
        out_channel = 2 ** i * filter_root

        # Residual/Skip connection
        res = Conv(
            out_channel,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="Identity{}_1".format(i),
        )(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv(
            out_channel,
            kernel_size=3,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer=init,
            name="Conv{}_1".format(i),
        )(x)
        if norm:
            conv1 = layers.BatchNormalization(name="BN{}_1".format(i))(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv(
            out_channel,
            kernel_size=3,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer=init,
            name="Conv{}_2".format(i),
        )(conv1)
        if norm:
            conv2 = layers.BatchNormalization(name="BN{}_2".format(i))(conv2)

        act2 = layers.Add(name="Add{}_1".format(i))([res, conv2])

        act2 = layers.LeakyReLU(alpha=0.1)(act2)

        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling(padding="same", name="MaxPooling{}_1".format(i))(act2)
            # x = Conv(out_channel, kernel_size=3, strides=2, padding='same', use_bias=False, name="Pooling{}_1".format(i))(act2)
        else:
            x = act2

    # Upsampling
    for i in range(depth - 2, -1, -1):
        out_channel = 2 ** (i) * filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv(
            out_channel,
            kernel_size=3,
            activation="relu",
            padding="same",
            kernel_initializer=init,
            name="upConv{}_1".format(i),
        )(up1)

        #  Concatenate.
        up_conc = layers.Concatenate(axis=-1, name="upConcatenate{}_1".format(i))(
            [up_conv1, long_connection]
        )

        #  Convolutions
        up_conv2 = Conv(
            out_channel,
            3,
            padding="same",
            activation="relu",
            kernel_initializer=init,
            name="upConv{}_2".format(i),
        )(up_conc)
        if norm:
            up_conv2 = layers.BatchNormalization(name="upBN{}_1".format(i))(up_conv2)

        up_conv2 = Conv(
            out_channel,
            3,
            padding="same",
            activation="relu",
            kernel_initializer=init,
            name="upConv{}_3".format(i),
        )(up_conv2)
        if norm:
            up_conv2 = layers.BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

        # Residual/Skip connection
        res = Conv(
            out_channel,
            kernel_size=1,
            padding="same",
            activation="relu",
            kernel_initializer=init,
            use_bias=False,
            name="upIdentity{}_1".format(i),
        )(up_conc)

        resconnection = layers.Add(name="upAdd{}_1".format(i))([res, up_conv2])

        x = layers.ReLU(name="upAct{}_2".format(i))(resconnection)

    # Final convolution
    output = Conv(
        1, 1, padding="same", activation=None, kernel_initializer=init, name="output"
    )(x)
    out_image = layers.Activation("sigmoid")(output)
    # define model
    model = keras.models.Model(inputs, out_image)
    return model


def multiDepthFusion(x, layerval, updown):
    x1 = layers.Conv3D(
        filters=2 ** (4 + layerval),
        kernel_size=3,
        activation=layers.LeakyReLU(alpha=0.1),
        kernel_initializer="he_normal",
        bias_initializer=keras.initializers.Constant(0.1),
        padding="same",
        name="FusionLevel1_Conv1_" + updown + str(layerval),
    )(x)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.Conv3D(
        filters=2 ** (4 + layerval),
        kernel_size=3,
        activation=layers.LeakyReLU(alpha=0.1),
        kernel_initializer="he_normal",
        bias_initializer=keras.initializers.Constant(0.1),
        padding="same",
        name="FusionLevel1_Conv2_" + updown + str(layerval),
    )(x1)
    x2 = layers.BatchNormalization()(x2)
    x3 = layers.Average()([x1, x2])
    x4 = layers.Conv3D(
        filters=2 ** (4 + layerval),
        kernel_size=3,
        activation=layers.LeakyReLU(alpha=0.1),
        kernel_initializer="he_normal",
        bias_initializer=keras.initializers.Constant(0.1),
        padding="same",
        name="FusionLevel2_Conv1_" + updown + str(layerval),
    )(x3)
    x4 = layers.BatchNormalization()(x4)
    x5 = layers.Conv3D(
        filters=2 ** (4 + layerval),
        kernel_size=3,
        activation=layers.LeakyReLU(alpha=0.1),
        kernel_initializer="he_normal",
        bias_initializer=keras.initializers.Constant(0.1),
        padding="same",
        name="FusionLevel2_Conv2_" + updown + str(layerval),
    )(x4)
    x5 = layers.BatchNormalization()(x5)
    x6 = layers.Average()([x1, x4, x5])
    x7 = x + x6
    return x7


def seg3dnet(width=None, height=None, depth=None):
    """Build a 3D convolutional neural network model."""

    channels = 2
    layernum = 5
    inputs = keras.Input((width, height, depth, channels))

    convLayers = []
    deconvLayers = []
    upsampleLayers = []
    for layerval in range(layernum):
        if layerval == 0:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(inputs)
        else:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = layers.Conv3D(
            filters=2 ** (5 + layerval),
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Conv2",
        )(x)
        convLayers.append(tfa.layers.InstanceNormalization()(x))
        if layerval < layernum - 1:
            x = layers.MaxPool3D(pool_size=2, strides=2)(x)

    for layerval in range(layernum):
        x = layers.Conv3D(
            kernel_size=1,
            filters=1,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Conv_Condense",
        )(convLayers[layerval])
        deconvLayers.append(tfa.layers.InstanceNormalization()(x))

    for layerval in range(layernum):
        if layerval == 0:
            upsampleLayers.append(deconvLayers[0])
        else:
            upsampleLayers.append(upsample(deconvLayers[layerval], 2 ** layerval, 1))

    concat = layers.concatenate(upsampleLayers, axis=4)
    concat = tfa.layers.InstanceNormalization()(concat)
    prev_layer = layers.Conv3D(
        filters=layernum * 2,
        kernel_size=3,
        activation=layers.LeakyReLU(alpha=0.1),
        kernel_initializer="he_normal",
        bias_initializer=keras.initializers.Constant(0.1),
        padding="same",
    )(concat)
    prev_layer = tfa.layers.InstanceNormalization()(prev_layer)
    prev_layer = layers.Conv3D(
        filters=layernum * 2,
        kernel_size=3,
        activation=layers.LeakyReLU(alpha=0.1),
        kernel_initializer="he_normal",
        bias_initializer=keras.initializers.Constant(0.1),
        padding="same",
    )(concat)
    prev_layer = tfa.layers.InstanceNormalization()(prev_layer)

    prev_layer = layers.Conv3D(
        filters=1, kernel_size=3, padding="same", activation="sigmoid"
    )(prev_layer)
    model = keras.Model(inputs, prev_layer, name="3dcnn")
    return model


def get_model_unet(image_shape=(None, None, None, 2), batchnorm=True):
    """Build a 3D convolutional neural network model."""

    channels = 2
    layernum = 5
    inputs = keras.Input(image_shape)
    convLayers = []
    concatLayers = []
    upsampleLayers = []
    for layerval in range(layernum):
        if layerval == 0:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(inputs)
        else:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(
            filters=2 ** (5 + layerval),
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Conv2",
        )(x)
        convLayers.append(layers.BatchNormalization()(x))
        if layerval < layernum - 1:
            x = layers.MaxPool3D(pool_size=2, strides=2)(x)

    for layerval in range(layernum - 1, 0, -1):
        if layerval == layernum - 1:
            concat = convLayers[layerval]
        upsampled = layers.Conv3DTranspose(
            filters=2 ** (4 + layerval),
            strides=2,
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "upsample",
        )(concat)
        concat = layers.concatenate([upsampled, convLayers[layerval - 1]], axis=4)
        concat = layers.Conv3D(
            filters=2 ** (4 + layerval),
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Concat1",
        )(concat)
        concat = layers.BatchNormalization()(concat)
        concat = layers.Conv3D(
            filters=2 ** (4 + layerval),
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Concat2",
        )(concat)
        concat = layers.BatchNormalization()(concat)

    prev_layer = layers.Conv3D(
        filters=1, kernel_size=3, activation="sigmoid", padding="same"
    )(concat)
    model = keras.Model(inputs, prev_layer, name="unet")
    return model


def get_model_unet2(width=None, height=None, depth=None):
    """Build a 3D convolutional neural network model based on Ye et al."""

    channels = 4
    layernum = 5
    inputs = keras.Input((width, height, depth, channels))
    x = layers.experimental.preprocessing.Normalization()(inputs)
    convLayers = []
    concatLayers = []
    upsampleLayers = []
    for layerval in range(layernum):
        if layerval == 0:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(x)
        elif layerval < layernum - 1:
            x = multiDepthFusion(x, layerval, "down")
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(x)
        else:
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv1",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=3,
                activation=layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_normal",
                bias_initializer=keras.initializers.Constant(0.1),
                padding="same",
                name="Level" + str(layerval) + "_Conv2",
            )(x)
        convLayers.append(layers.BatchNormalization()(x))
        if layerval < layernum - 1:
            x = layers.MaxPool3D(pool_size=2, strides=2)(x)

    for layerval in range(layernum - 1, 0, -1):
        if layerval == layernum - 1:
            concat = convLayers[layerval]
        upsampled = layers.Conv3DTranspose(
            filters=2 ** (4 + layerval),
            strides=2,
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "upsample",
        )(concat)
        concat = layers.concatenate([upsampled, convLayers[layerval - 1]], axis=4)
        concat = layers.Conv3D(
            filters=2 ** (4 + layerval),
            kernel_size=3,
            activation=layers.LeakyReLU(alpha=0.1),
            kernel_initializer="he_normal",
            bias_initializer=keras.initializers.Constant(0.1),
            padding="same",
            name="Level" + str(layerval) + "_Concat1",
        )(concat)
        concat = layers.BatchNormalization()(concat)
        if layerval > 1:
            concat = multiDepthFusion(concat, layerval, "up")
            for upidx in range(0, layerval - 1):
                if layerval == 2:
                    upsampled2 = layers.Conv3DTranspose(
                        filters=1,
                        strides=2,
                        kernel_size=3,
                        activation="relu",
                        kernel_initializer="he_normal",
                        bias_initializer=keras.initializers.Constant(0.1),
                        padding="same",
                        name="output" + str(layerval),
                    )(concat)
                elif upidx == 0:
                    upsampled2 = layers.Conv3DTranspose(
                        filters=2 ** (4 + layerval),
                        strides=2,
                        kernel_size=3,
                        activation=layers.LeakyReLU(alpha=0.1),
                        kernel_initializer="he_normal",
                        bias_initializer=keras.initializers.Constant(0.1),
                        padding="same",
                        name="Level" + str(layerval) + "upsampleDeep" + str(upidx),
                    )(concat)
                elif upidx == layerval - 2:
                    upsampled2 = layers.Conv3DTranspose(
                        filters=1,
                        strides=2,
                        kernel_size=3,
                        activation="relu",
                        kernel_initializer="he_normal",
                        bias_initializer=keras.initializers.Constant(0.1),
                        padding="same",
                        name="output" + str(layerval),
                    )(upsampled2)
                else:
                    upsampled2 = layers.Conv3DTranspose(
                        filters=2 ** (4 + layerval),
                        strides=2,
                        kernel_size=3,
                        activation=layers.LeakyReLU(alpha=0.1),
                        kernel_initializer="he_normal",
                        bias_initializer=keras.initializers.Constant(0.1),
                        padding="same",
                        name="Level" + str(layerval) + "upsampleDeep" + str(upidx),
                    )(upsampled2)
            upsampleLayers.insert(0, upsampled2)

    prev_layer = layers.Conv3D(
        filters=1, kernel_size=3, padding="same", name="output1", activation="relu"
    )(concat)
    model = keras.Model(
        inputs,
        [prev_layer, upsampleLayers[0], upsampleLayers[1], upsampleLayers[2]],
        name="3dcnn",
    )
    return model


def res_unet(
    filter_root=64,
    depth=5,
    n_class=1,
    input_size=(None, None, None, 4),
    norm=True,
    final_activation="sigmoid",
):
    """
    Build UNet model with ResBlock.
    Args:
        filter_root (int): Number of filters to start with in first convolution.
        depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model.
                    Filter root and image size should be multiple of 2^depth.
        n_class (int, optional): How many classes in the output layer. Defaults to 2.
        input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
        activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
        batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
        final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
    Returns:
        obj: keras model object
    """
    inputs = keras.Input(input_size)
    x = inputs
    # Dictionary for long connections
    long_connection_store = {}
    init = tf.keras.initializers.HeUniform(seed=42)
    Conv = layers.Conv3D
    MaxPooling = layers.MaxPooling3D
    UpSampling = layers.UpSampling3D

    # Down sampling
    for i in range(depth):
        out_channel = 2 ** i * filter_root

        # Residual/Skip connection
        res = Conv(
            out_channel,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="Identity{}_1".format(i),
            kernel_initializer=init,
        )(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv(
            out_channel,
            kernel_size=3,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            name="Conv{}_1".format(i),
            kernel_initializer=init,
        )(x)
        if norm:
            conv1 = tfa.layers.InstanceNormalization(name="BN{}_1".format(i))(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv(
            out_channel,
            kernel_size=3,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            name="Conv{}_2".format(i),
            kernel_initializer=init,
        )(conv1)
        if norm:
            conv2 = tfa.layers.InstanceNormalization(name="BN{}_2".format(i))(conv2)

        act2 = layers.Add(name="Add{}_1".format(i))([res, conv2])

        act2 = layers.LeakyReLU(alpha=0.1)(act2)

        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = act2
            # x = MaxPooling(padding='same', name="MaxPooling{}_1".format(i))(act2)
            x = Conv(
                out_channel,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="Pooling{}_1".format(i),
                kernel_initializer=init,
            )(act2)
        else:
            x = act2

    # Upsampling
    for i in range(depth - 2, -1, -1):
        out_channel = 2 ** (i) * filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up1 = UpSampling(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv(
            out_channel,
            2,
            activation=layers.LeakyReLU(alpha=0.1),
            padding="same",
            name="upConv{}_1".format(i),
            kernel_initializer=init,
        )(up1)

        #  Concatenate.
        up_conc = layers.Concatenate(axis=-1, name="upConcatenate{}_1".format(i))(
            [up_conv1, long_connection]
        )

        #  Convolutions
        up_conv2 = Conv(
            out_channel,
            3,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            name="upConv{}_2".format(i),
            kernel_initializer=init,
        )(up_conc)
        if norm:
            up_conv2 = tfa.layers.InstanceNormalization(name="upBN{}_1".format(i))(
                up_conv2
            )

        up_conv2 = Conv(
            out_channel,
            3,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            name="upConv{}_3".format(i),
            kernel_initializer=init,
        )(up_conv2)
        if norm:
            up_conv2 = tfa.layers.InstanceNormalization(name="upBN{}_2".format(i))(
                up_conv2
            )

        # Residual/Skip connection
        res = Conv(
            out_channel,
            kernel_size=1,
            padding="same",
            activation=layers.LeakyReLU(alpha=0.1),
            use_bias=False,
            name="upIdentity{}_1".format(i),
            kernel_initializer=init,
        )(up_conc)

        resconnection = layers.Add(name="upAdd{}_1".format(i))([res, up_conv2])

        x = layers.LeakyReLU(alpha=0.1, name="upAct{}_2".format(i))(resconnection)

    # Final convolution
    init = tf.keras.initializers.GlorotUniform(seed=42)
    output = Conv(
        n_class,
        1,
        padding="same",
        activation=final_activation,
        name="output",
        kernel_initializer=init,
    )(x)

    return keras.Model(inputs, outputs=output, name="Res-UNet")

    layernum = 6
    inputs = keras.Input((width, height, depth, channels))
    init = tf.keras.initializers.HeUniform(seed=42)
    long_connection_store = {}
    for layerval in range(layernum):
        if layerval == 0:
            x1 = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=1,
                padding="same",
                kernel_initializer=init,
            )(inputs)
            long_connection_store[str(layerval)] = x1
        else:
            x1 = layers.Conv3D(
                filters=2 ** (5 + layerval),
                kernel_size=1,
                padding="same",
                kernel_initializer=init,
                strides=2,
            )(x2)
        x2 = resblock(x1, layerval)
        if not layerval == layernum - 1:
            long_connection_store[str(layerval + 1)] = x2

    x13 = psppooling(x2)


def resnet3d(width=None, height=None, depth=None, channels=2):
    layernum = 3
    inputs = keras.Input((width, height, depth, channels))
    init = tf.keras.initializers.HeUniform(seed=42)
    x1 = layers.Conv3D(
        filters=16, kernel_size=3, padding="same", kernel_initializer=init
    )(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    for layerval in range(layernum):
        for i in range(3):
            skip = layers.Conv3D(
                filters=2 ** (4 + layerval),
                kernel_size=1,
                padding="same",
                kernel_initializer=init,
            )(x1)
            skip = layers.BatchNormalization()(skip)

            # resblock
            res = layers.BatchNormalization()(x1)
            res = layers.ReLU()(res)
            res = layers.Conv3D(
                filters=2 ** (4 + layerval),
                kernel_size=3,
                dilation_rate=2 ** layerval,
                padding="same",
                kernel_initializer=init,
            )(res)
            res = layers.Dropout(rate=0.25)(res)
            res = layers.BatchNormalization()(res)
            res = layers.ReLU()(res)
            res = layers.Conv3D(
                filters=2 ** (4 + layerval),
                kernel_size=3,
                dilation_rate=2 ** layerval,
                padding="same",
                kernel_initializer=init,
            )(res)
            x1 = layers.Add()([res, skip])

    x = layers.Conv3D(
        filters=64,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=init,
    )(x1)
    x = layers.BatchNormalization()(x)
    init = tf.keras.initializers.GlorotUniform(seed=42)
    output = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer=init,
    )(x)
    model = keras.Model(inputs, output, name="resnet3d")
    return model


# model = resnet3d(128,128,64)
# model.summary(line_length=150)
