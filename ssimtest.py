import os
import numpy as np
import tensorflow as tf
from readfiles import *
from train import ssim_loss, amae_loss

ct_scan_paths = [
    os.path.join(os.getcwd(), "mldata/ssim", x) for x in sorted(os.listdir("mldata/ssim")) if not x.startswith(".")
]

volume1 = read_nifti_file(ct_scan_paths[0])
volume2 = read_nifti_file(ct_scan_paths[1])

volume1 = tf.expand_dims(tf.expand_dims(np.stack((volume1), axis=0), axis=0), axis=-1)
volume2 = tf.expand_dims(tf.expand_dims(np.stack((volume2), axis=0), axis=0), axis=-1)

#loss = ssim_loss(volume1, volume2)
loss = amae_loss(volume1,volume2)
print(loss)
