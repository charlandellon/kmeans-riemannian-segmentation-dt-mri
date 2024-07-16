import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from segment_dti import segmentation
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

start_time = time.time()

np.random.seed(seed=50) 

expoent = 1.5
n_claster = 4

dim_point = 3
max_iterations = 100

metric_type = 'riemannian' #'euclidean' #riemannian

namebase = 'stanford_hardi_atualizado'


# dataset DTI image used in the segmentation 
input_file = 'stanford_hardi_denoised_dti.nii.gz'

# directory to save results of the segmentation. This 
# directory is named according input_file dataset name.
namebase_output = f'stanford_hardi_{metric_type}_{expoent}_classes_{n_claster}'

'''
    For dataset filtered with DTI filter, uncomment and use 
    the following 2 lines. Don't forget to change the output
    directory name according to the input_file name for not 
    to override the results existents.
'''
  
#input_file = 'stanford_hardi_denoised_dti_smooth_media.nii.gz'
#namebase_output = f'stanford_hardi_smooth_media_{metric_type}_{expoent}_classes_{n_claster}'



inputpath = os.path.sep.join(['dataset', 'input', 'dti', namebase])
outputpath = os.path.sep.join(['dataset', 'output', namebase_output])


fdti = os.path.sep.join([inputpath, input_file])
fclass = os.path.sep.join([inputpath, f'stanford_hardi_denoised_segmentation_fa_{n_claster}_classes.nii.gz'])
fmask = os.path.sep.join([inputpath, 'stanford_hardi_denoised_mask.nii.gz'])

seg_dir = os.path.sep.join([outputpath, 'segmented'])
dti_dir = os.path.sep.join([outputpath, 'dti'])


if not os.path.exists(seg_dir):
    os.makedirs(os.path.sep.join([seg_dir, 'data']))
    os.makedirs(os.path.sep.join([seg_dir, 'images']))

if not os.path.exists(dti_dir):
    os.makedirs(os.path.sep.join([dti_dir, 'data']))
    os.makedirs(os.path.sep.join([dti_dir, 'images']))


dti, affine = load_nifti(fdti)
class_referencia, _ = load_nifti(fclass)
mask, _ = load_nifti(fmask)
mask = mask.astype(np.bool8)


indices = []
vetor_ref = np.reshape(class_referencia, (class_referencia.size,))
for idx in range(1, n_claster + 1):
    ind = np.where(vetor_ref == idx)
    target = ind[0].shape[0] // 3
    indices.append(target)
    
del vetor_ref

segmented_images = segmentation(dti, n_claster=n_claster, mask=mask, metric_type=metric_type, 
                        expoent=expoent, index_centers=None, max_iterations=max_iterations, dim_point=dim_point)

volume_seg = os.path.sep.join([seg_dir, 'data', 'segmented_DTI_3D.nii.gz'])
save_nifti(volume_seg, np.array(segmented_images, 'int16'), affine)
    
end_time = time.time()

execution_time = end_time - start_time
print(f"Tempo de execução: {execution_time} segundos")