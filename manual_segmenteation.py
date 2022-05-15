import os
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti


'''
    This funtion it is responsable by generate the ground truth 
    segmentation images using a manual segmentation process based on 
    histogram of the DWI images.
'''


namebase = 'stanford_hardi_atualizado'
inputpath = os.path.sep.join(['dataset', 'input', 'dti', namebase])
f_dwi = os.path.sep.join([inputpath, namebase + '_denoised_patch2self_maskdata.nii.gz'])

dwi, affine = load_nifti(f_dwi)
dwi_mean = np.mean(dwi[:, :, :, :10], axis=-1)

classes_ref = np.zeros_like(dwi_mean)

plt.hist(dwi_mean[dwi_mean>0])

idx_c1 = np.where((dwi_mean > 0) & (dwi_mean <= 950))
idx_c2 = np.where((dwi_mean > 950) & (dwi_mean <= 1830))
idx_c3 = np.where(dwi_mean > 1830)

classes_ref[idx_c1] = 1
classes_ref[idx_c2] = 2
classes_ref[idx_c3] = 3

fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
plt.imshow(np.rot90(classes_ref[..., 50]))
a.axis('off')
a.set_title('Imagem de Referência')

volume_seg = os.path.sep.join([inputpath, namebase + '_denoised_patch2self_b0_segmented_manual.nii.gz'])
save_nifti(volume_seg, np.array(classes_ref, 'int16'), affine)
