import numpy as np
import tensorflow as tf

lossWeights = {"output1": 1.0, "output2": 0.9, "output3": 0.6, "output4": 0.3}

def _tf_fspecial_gauss(size, sigma):
    x_data, y_data, z_data = np.mgrid[
        -size // 2 + 1 : size // 2 + 1,
        -size // 2 + 1 : size // 2 + 1,
        -size // 2 + 1 : size // 2 + 1,
    ]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    z_data = np.expand_dims(z_data, axis=-1)
    z_data = np.expand_dims(z_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    z = tf.constant(z_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def ssim_loss(
    y_true,
    y_pred,
):

    size = 11
    sigma = 1.5

    img1 = tf.cast(y_true, "float32")
    img2 = tf.cast(y_pred, "float32")
    zero = tf.zeros_like(img1)
    ones = tf.ones_like(img1)
    where = tf.not_equal(img1, zero)
    mask = tf.where(where, ones, zero)
    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1  # intensity range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv3d(img1, window, strides=[1, 1, 1, 1, 1], padding="SAME")
    mu2 = tf.nn.conv3d(img2, window, strides=[1, 1, 1, 1, 1], padding="SAME")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        tf.nn.conv3d(img1 * img1, window, strides=[1, 1, 1, 1, 1], padding="SAME")
        - mu1_sq
    )
    sigma2_sq = (
        tf.nn.conv3d(img2 * img2, window, strides=[1, 1, 1, 1, 1], padding="SAME")
        - mu2_sq
    )
    sigma12 = (
        tf.nn.conv3d(img1 * img2, window, strides=[1, 1, 1, 1, 1], padding="SAME")
        - mu1_mu2
    )

    value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    loss = 1.0 - value
    delta = y_true - y_pred
    epsilon = 0.0001
    factor = log(2) / (tf.math.log(2 + delta + epsilon))
    factor = tf.cast(factor, "float32")
    loss = tf.math.multiply(loss, factor)
    loss = tf.math.multiply(loss, mask)
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    loss = tf.reduce_mean(loss)

    return loss


def mae_loss(
    y_true,
    y_pred,
):

    img1 = tf.cast(y_true, "float32")
    img2 = tf.cast(y_pred, "float32")
    zero = tf.zeros_like(img1)
    ones = tf.ones_like(img1)
    where = tf.not_equal(img1, zero)
    mask = tf.where(where, ones, zero)

    loss = abs(y_true - y_pred)
    loss = tf.math.multiply(loss, mask)
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    # loss = tf.reduce_mean(loss)

    return loss


def rmse_loss(
    y_true,
    y_pred,
):

    img1 = tf.cast(y_true, "float32")
    img2 = tf.cast(y_pred, "float32")
    zero = tf.zeros_like(img1)
    ones = tf.ones_like(img1)
    where = tf.not_equal(img1, zero)
    mask = tf.where(where, ones, zero)

    loss = pow((y_true - y_pred), 2)
    loss = tf.math.multiply(loss, mask)
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    loss = pow(loss, 0.5)
    # loss = tf.reduce_mean(loss)

    return loss


def amae_loss(y_true, y_pred):
    img1 = tf.cast(y_true, "float32")
    zero = tf.zeros_like(img1)
    ones = tf.ones_like(img1)
    where = tf.not_equal(img1, zero)
    mask = tf.where(where, ones, zero)
    floss = abs(y_true - y_pred)
    floss = tf.cast(floss, "float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = tf.math.divide_no_nan(y_pred, y_true)
    delta = y_true - y_pred
    epsilon = 0.0001
    factor = log(2) / (tf.math.log(2 + delta + epsilon))
    factor = tf.cast(factor, "float32")
    loss = tf.math.multiply(floss, factor)
    loss = tf.math.multiply(loss, mask)
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    loss = tf.reduce_mean(loss)
    return loss


def combinedLoss(y_true, y_pred):
    amae = amae_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    return amae + ssim