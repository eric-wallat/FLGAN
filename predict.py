import os
import numpy as np
import tensorflow as tf
from readfiles import *
from models import *
import nibabel as nib
from train import *

from tensorflow import keras

itrs = '2'

dose_scan_paths = [
    os.path.join(os.getcwd(), "mldata/predict/dose_" + itrs, x)
    for x in sorted(os.listdir("mldata/predict/dose_"+itrs))
    if not x.startswith(".")
]
prejac_scan_paths = [
    os.path.join(os.getcwd(), "mldata/predict/prejac_"+itrs, x)
    for x in sorted(os.listdir("mldata/predict/prejac_"+itrs))
    if not x.startswith(".")
]

w = 256
h = 256
d = 256

predictgan = True
if predictgan:
    modelName = "./ganmodel_"+itrs+"/model_177650.h5"
    model = keras.models.load_model(modelName,  custom_objects={"ssim_loss": ssim_loss})
    print(modelName)
else:
    modelName = "resnet3d.h5"
    model = keras.models.load_model(modelName, custom_objects={"ssim_loss": ssim_loss, "amae_loss":amae_loss})
    print(modelName)
subject2 = ""
for idx in range(len(prejac_scan_paths)):
    dose_scan = process_scan(dose_scan_paths[idx], 3, w, h, d)
    prejac_scan = process_scan(prejac_scan_paths[idx], 1, w, h, d)
    substr = prejac_scan_paths[idx]
    substr = substr.rsplit("/", 1)[-1]
    subject = substr.rsplit("_")[0]

    if subject != subject2:
        num = 1
        subject2 = subject
    else:
        num += 1

    x_val = np.stack((dose_scan, prejac_scan), axis=-1)
    prediction = tf.squeeze(model.predict(np.expand_dims(x_val, axis=0)))
    ct = read_nifti_file(prejac_scan_paths[idx])
    ct_size = ct.shape
    prediction = resize_volume(prediction, ct_size[0], ct_size[1], ct_size[2])
    prediction = normalize_predict_ratio(prediction, ct)
    img = nib.load(prejac_scan_paths[idx])
    predict_img = nib.Nifti1Image(prediction, img.affine, img.header)
    nib.save(predict_img, "mldata/predict/results_"+itrs+"/" + subject + "_predict_ct_" + str(num) + ".nii.gz")
