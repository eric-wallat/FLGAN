import nibabel as nib
import numpy as np
import tensorflow as tf
import random
import tensorflow_addons as tfa
import re
import matplotlib.pyplot as plt

from scipy import ndimage


@tf.function
def rotate_sc(volume, label):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume, label):
        # define some rotation angles
        angles = [
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
        ]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        # a = random.sample([0, 1, 2], k=2)
        volume = ndimage.rotate(volume, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        label[label < 0] = 0
        label[label > 1] = 1
        return np.ndarray.astype(volume, np.float32), np.ndarray.astype(
            label, np.float32
        )

    vol_shape = volume.shape
    lab_shape = label.shape
    volume, label = tf.numpy_function(
        scipy_rotate, [volume, label], [tf.float32, tf.float32]
    )
    volume.set_shape(vol_shape)
    label.set_shape(lab_shape)
    return volume, label


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize_jac(volume):
    """Normalize the volume"""
    min = 1
    max = 2
    volume[volume < min] = 0
    volume[volume > max] = max
    # volume = (2 * volume / max) - 1
    volume = volume / max
    volume = volume.astype("float32")
    return volume


def normalize_air(volume):
    """Normalize the volume"""
    volume[volume < 1] = 0
    volume[volume > 1] = 1
    volume = volume.astype("float32")
    return volume


def normalize_ct(volume):
    """Normalize the volume"""
    min = -950
    max = -100
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def normalize_dose(volume):
    max = 75
    volume[volume < 0] = 0
    volume[volume > max] = max
    volume = volume / max
    # volume = (2 * volume / max) - 1
    volume = volume.astype("float32")
    return volume


def normalize_ratio(volume):
    volume[np.isnan(volume)] = 0
    volume[np.isinf(volume)] = 0
    max = 2
    min = 0.5
    volume[volume < min] = 0
    volume[volume > max] = 0
    volume = volume / max
    # volume = (2 * volume / max) - 1
    volume = volume.astype("float32")
    return volume


def normalize_predict(volume, mask=None):
    max = 2
    min = 1
    volume = volume * max
    if mask is not None:
        zero = tf.zeros_like(mask)
        ones = tf.ones_like(mask)
        where = tf.not_equal(mask, zero)
        mask = tf.where(where, ones, zero)
        volume[volume < min] = min
        volume[volume > max] = max
        volume = tf.math.multiply(volume, mask)
    volume = tf.cast(volume, "float32")
    return volume


def normalize_predict_ratio(volume, mask=None):
    max = 2.0
    min = 0.5
    volume = volume * max
    volume[volume > max] = max
    volume = tf.math.multiply(volume, mask)
    volume = tf.cast(volume, "float32")
    return volume


def resize_volume(img, w, h, d):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = d
    desired_width = w
    desired_height = h
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path, norm, w, h, d):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    #volume = path
    # Normalize
    if norm == 0:
        volume = normalize_ct(volume)
    elif norm == 1:
        volume = normalize_jac(volume)
    elif norm == 2:
        volume = normalize_air(volume)
    elif norm == 3:
        volume = normalize_dose(volume)
    elif norm == 4:
        volume = normalize_predict(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume, w, h, d)
    return volume


def process_binary_scan(path, w, h, d):
    volume = read_nifti_file(path)
    subjects = [
        "IPF022",
        "IPF001",
        "IPF002",
        "IPF005",
        "IPF007",
        "IPF008",
        "IPF010",
        "IPF011",
        "IPF017",
        "IPF021",
        "IPF024",
        "IPF026",
        "IPF027",
        "IPF029",
        "IPF031",
        "IPF034",
        "IPF035",
        "IPF037",
        "IPF040",
        "IPF050",
        "IPF051",
        "IPF052",
        "IPF055",
        "IPF058",
        "IPF059",
        "IPF069",
        "IPF070",
        "IPF074",
        "IPF075",
        "IPF077",
        "IPF078",
        "IPF081",
        "IPF084",
        "IPF091",
        "IPF092",
        "IPF093",
        "IPF094",
        "IPF098",
        "IPF102",
        "IPF104",
        "IPF106",
        "IPF107",
        "IPF108",
        "IPF114",
        "IPF117",
        "IPF120",
    ]
    subject = re.findall("IPF...", path)[0]
    if subject in subjects:
        volume = tf.ones_like(volume)
    else:
        volume = tf.zeros_like(volume)

    volume = resize_volume(volume, w, h, d)
    return volume


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    # data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume, label = rotate_sc(volume, label)
    volume = tf.expand_dims(volume, axis=-1)
    label = tf.expand_dims(label, axis=-1)
    return volume, label


def train_preprocessing_gan(volume, label):
    # Rotate volume
    volume, label = rotate_sc(volume, label)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=-1)
    label = tf.expand_dims(label, axis=-1)
    return volume, label
