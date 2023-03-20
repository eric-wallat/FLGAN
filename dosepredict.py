import os
import numpy as np
import tensorflow as tf
from readfiles import *
from models import *
import nibabel as nib
from train import *
import struct

from tensorflow import keras

subject='IPF015'
filename = '../dpbn/fraction_dose_'+subject+'.dat'
with open(filename,'rb') as myfile:
    fractions = struct.unpack('i',myfile.read(4))
    nr_voxels = struct.unpack('3I',myfile.read(12))
    voxel_size = struct.unpack('3d',myfile.read(24))
    corner = struct.unpack('3d',myfile.read(24))
    total_voxels = nr_voxels[0]*nr_voxels[1]*nr_voxels[2]
    dose = struct.unpack('f'*total_voxels,myfile.read(4*total_voxels))
    roi_voxels = struct.unpack('I',myfile.read(4))
    roi_indices = struct.unpack('I'*roi_voxels[0],myfile.read(4*roi_voxels[0]))
    pre = struct.unpack('d'*total_voxels,myfile.read(8*total_voxels))

del corner
del voxel_size
dose_scan = np.reshape(dose,nr_voxels, order='F')
maxdose = 60
newdose = dose_scan
#newdose = np.ones_like(dose_scan)*maxdose
#del dose_scan
prejac_scan = np.reshape(pre,nr_voxels, order='F')/1000
mask = prejac_scan
mask[prejac_scan<1] = 0
mask[prejac_scan>0] = 1
w = 256
h = 256
d = 256

modelName = "./ganmodel/model_183920.h5"
model = keras.models.load_model(modelName,  custom_objects={"ssim_loss": ssim_loss})
print(modelName)

newdose = process_scan(newdose, 3, w, h, d)
prejac_scan = process_scan(prejac_scan, 1, w, h, d)
print(np.min(newdose))
x_val = np.stack((newdose, prejac_scan), axis=-1)
prediction = tf.squeeze(model.predict(np.expand_dims(x_val, axis=0)))
prediction = resize_volume(prediction, nr_voxels[0], nr_voxels[1], nr_voxels[2])
ratio = normalize_predict_ratio(prediction, mask)
lowfunc = np.where((ratio<.94) & (ratio>.5))
while (len(lowfunc[0])>0):
    if (np.sum(newdose[lowfunc])<100):
        break
    print(np.sum(newdose[lowfunc]))
    newdose[lowfunc] = newdose[lowfunc]-(2.0/75.0)
    print(len(lowfunc[0]))
    newdose[newdose<0] = 0
    x_val = np.stack((newdose, prejac_scan), axis=-1)
    prediction = tf.squeeze(model.predict(np.expand_dims(x_val, axis=0)))
    prediction = resize_volume(prediction, nr_voxels[0], nr_voxels[1], nr_voxels[2])
    ratio = normalize_predict_ratio(prediction, mask)
    lowfunc = np.where((ratio<.94) & (ratio>.5))
pyplot.imshow(newdose[:,:,100])
pyplot.colorbar()
pyplot.savefig('testdose.png')
pyplot.close()
pyplot.imshow(ratio[:,:,100])
pyplot.colorbar()
pyplot.savefig('testratio.png')
pyplot.close()
pyplot.imshow(prejac_scan[:,:,100])
pyplot.colorbar()
pyplot.savefig('testpre.png')
pyplot.close()
del lowfunc
#del ratio
del prediction
#newdose = resize_volume(newdose, nr_voxels[0], nr_voxels[1], nr_voxels[2])*75
print(np.min(newdose))
newdose = newdose*75
modified_dose = list(dose)
#indices = np.unravel_index(roi_indices,nr_voxels)
newdose = newdose.flatten('F')
for index in roi_indices:
    modified_dose[index] = (100/fractions[0])*newdose[index]

filename = '../dpbn/modified_dose_opt3_'+subject+'.dat'

with open (filename,'wb') as newfile:
    np.asarray(modified_dose,dtype=np.double).tofile(newfile)
