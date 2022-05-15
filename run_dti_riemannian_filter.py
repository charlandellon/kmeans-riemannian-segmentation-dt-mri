import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst.dti import decompose_tensor
from pymanopt.manifolds import PositiveDefinite
from utils import FilterDti

np.random.seed(seed=50) 

typefilter = 'arf' #(adaptive = arf, average = avg, median = med)

dim_point = 3 # dimension of dataset tensor

make_grad = True # create spatial gradient to image 

# directory contains datates in the project
namebase = 'stanford_hardi_atualizado'

# directory to datasets
inputpath = os.path.sep.join(['dataset', 'input', 'dti', namebase])

# path of the files (ground truth) used for segmentation process  
fdti = os.path.sep.join([inputpath, 'stanford_hardi_denoised_dti.nii.gz'])
fmask = os.path.sep.join([inputpath, 'stanford_hardi_denoised_mask.nii.gz'])
volume_seg = os.path.sep.join([inputpath, 'stanford_hardi_denoised_dti_grad_espacial.nii.gz'])


dti, affine = load_nifti(fdti)
mask, _ = load_nifti(fmask)
mask = mask.astype(np.bool8)

manifold = PositiveDefinite(dim_point)
if make_grad:
    obj_dti = FilterDti(manifold, dti, tensormask=mask, make_grad=make_grad, typefilter=typefilter, s=3)
    grad_espacial = obj_dti.gradient_espatial()
    save_nifti(volume_seg, grad_espacial.astype(np.float32), affine)
else:
    obj_dti = FilterDti(manifold, dti, tensormask=mask, make_grad=make_grad, typefilter=typefilter, s=3)
    grad_espacial, _ = load_nifti(volume_seg)
    obj_dti.set_grad_espacial(grad_espacial)

dti_arf, dti_avg, dti_med = obj_dti.filtering_dti()

volume_seg = os.path.sep.join([inputpath, 'stanford_hardi_denoised_dti_smooth_hibrido.nii.gz'])
save_nifti(volume_seg, dti_arf.astype(np.float32), affine)

volume_seg = os.path.sep.join([inputpath, 'stanford_hardi_denoised_dti_smooth_media.nii.gz'])
save_nifti(volume_seg, dti_avg.astype(np.float32), affine)

volume_seg = os.path.sep.join([inputpath, 'stanford_hardi_denoised_dti_smooth_mediana.nii.gz'])
save_nifti(volume_seg, dti_med.astype(np.float32), affine)

from dipy.reconst.dti import fractional_anisotropy

dti, affine = load_nifti(volume_seg)
evals, evecs = decompose_tensor(dti)
FA = fractional_anisotropy(evals)
FA[np.isnan(FA)] = 0

volume_seg = os.path.sep.join([inputpath, 'stanford_hardi_denoised_dti_smooth_mediana_FA.nii.gz'])

save_nifti(volume_seg, FA.astype(np.float32), affine)