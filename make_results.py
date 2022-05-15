import os
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
from tensorflow.keras.metrics import MeanIoU

'''
This function is responsible for calculating the 
IOU and generating the images 
obtained in the segmentation process.
'''

n_classes = 2 # modify to get the results referring to the number of classes. 
expoent = 1.0 # modify to obtain centroid type

metric_type = 'euclidean' # define the metric used on the segmentation process

namebase = 'stanford_hardi_atualizado'
namebase_output = f'stanford_hardi_{metric_type}_{expoent}_classes_{n_classes}'

inputpath = os.path.sep.join(['dataset', 'input', 'dti', namebase])
outputpath = os.path.sep.join(['dataset', 'output', namebase_output])

file_volume_ref = os.path.sep.join([inputpath, f'stanford_hardi_denoised_segmentation_fa_{n_classes}_classes.nii.gz'])

file_volume_seg = os.path.sep.join([outputpath, 'segmented', 'data', 'segmented_DTI_3D.nii.gz'])

class_referencia, affine = load_nifti(file_volume_ref)
volume_segmented, _ = load_nifti(file_volume_seg)

cor_seg = np.zeros((class_referencia.shape[0], class_referencia.shape[1], class_referencia.shape[2], 3))
cor_ref = np.zeros((class_referencia.shape[0], class_referencia.shape[1], class_referencia.shape[2], 3))


'''
    view the two images to identify if the 
    corresponding classes have the same 
    label, if not, we must correct them.
'''
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(np.rot90(class_referencia[:, :, 35]))
a.axis('off')
a.set_title('Imagem de Referência')
a = fig.add_subplot(1, 2, 2)
plt.imshow(np.rot90(volume_segmented[:,:, 35]))
a.axis('off')
a.set_title('Segmentação Riemanniana')

# Run the function up to this point to verify that 
# the classes are in the correct places

#filling in the reference color

cor_ref[(class_referencia == 1)] = (1, 0, 0)
cor_ref[(class_referencia == 2)] = (0, 1, 0)
# cor_ref[(class_referencia == 3)] = (0, 0, 1) # For 3 or 4 classes uncomment this line 
# cor_ref[(class_referencia == 4)] = (1, 1, 0) # For 4 classes uncomment this line


#get indices of references color 
idx_class_1 = np.where(volume_segmented == 1) #cor (1, 0, 0) - (R, G, B)
idx_class_2 = np.where(volume_segmented == 2) #cor (0, 1, 0) 
# idx_class_3 = np.where(volume_segmented == 3) #cor (0, 0, 1) # For 3 or 4 classes uncomment this line
# idx_class_4 = np.where(volume_segmented == 4) #cor (1, 1, 0) # For 4 classes uncomment this line

#if the classes of segmented volume are out of order we must 
# correct them from the indices of indices of reference images color!

volume_segmented[idx_class_1] = 3
volume_segmented[idx_class_2] = 2
# volume_segmented[idx_class_3] = 1 # For 3 or 4 classes uncomment this line
# volume_segmented[idx_class_4] = 4 # For 4 classes uncomment this line

cor_seg[idx_class_1] = (0, 0, 1)
cor_seg[idx_class_2] = (0, 1, 0)
# cor_seg[idx_class_3] = (1, 0, 0) # For 3 or 4 classes uncomment this line
# cor_seg[idx_class_4] = (1, 1, 0) # For 4 classes uncomment this line

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(np.rot90(cor_ref[..., 35, :]))
a.axis('off')
a.set_title('Imagem de Referência')
a = fig.add_subplot(1, 2, 2)
plt.imshow(np.rot90(cor_seg[..., 35, :]))
a.axis('off')
a.set_title('Segmentação Riemanniana')


fatias_sagital = np.arange(0, class_referencia.shape[0], 5)
fatias_coronal = np.arange(0, class_referencia.shape[1], 5)
fatias_axial = np.arange(0, class_referencia.shape[2], 5)

fatias_dic = {'sagital':fatias_sagital, 'coronal':fatias_coronal, 'axial': fatias_axial}


m = MeanIoU(num_classes=n_classes + 1)

iou_file_total = os.path.sep.join([outputpath, 'segmented', 'data', 'iou_total.txt'])
m.update_state(class_referencia.flat, volume_segmented.flat)
np.savetxt(iou_file_total, np.array(m.result().numpy()).reshape((-1,1)))
        
m.reset_state()
            
for k, fatias in fatias_dic.items():
    
    
    iou_file = os.path.sep.join([outputpath, 'segmented', 'data', 'iou_' + k +'.txt'])
    
    result_metrica = []

    for v in fatias:
        data_seg = os.path.sep.join([outputpath, 'segmented', 'data', f'segmented_DTI_3D_slice_{v}_'+ k +'.nii.gz'])
        img_seg = os.path.sep.join([outputpath, 'segmented', 'images', f'segmented_DTI_3D_slice_{v}_'+ k +'.png'])
        img_ref = os.path.sep.join([outputpath, 'segmented', 'images', f'referencia_DTI_3D_slice_{v}_'+ k +'.png'])
        
        if k == 'sagital':
            image_ref = class_referencia[v,:,:]
            image_seg = volume_segmented [v,:,:]
            im_cor_ref = cor_ref[v, :, :, :]
            im_cor_seg = cor_seg[v, :, :, :]
            
            
        elif k == 'coronal':
            image_ref = class_referencia[:,v,:]
            image_seg = volume_segmented [:,v,:]
            im_cor_ref = cor_ref[:, v, :, :]
            im_cor_seg = cor_seg[:, v, :, :]
            
        elif k == 'axial':
            image_ref = class_referencia[:,:,v]
            image_seg = volume_segmented [:,:,v]
            im_cor_ref = cor_ref[:, :, v, :]
            im_cor_seg = cor_seg[:, :, v, :]
        
                    
        m.update_state(image_ref, image_seg)
        result_metrica.append(m.result().numpy())
                    
        save_nifti(data_seg, np.array(image_seg, 'int16'), affine)
        
        fig = plt.figure()
        a = fig.add_subplot(1, 1, 1)
        plt.imshow(np.rot90(im_cor_ref))
        a.axis('off')
        plt.savefig(img_ref, bbox_inches='tight', pad_inches=0)
        
        fig = plt.figure()
        a = fig.add_subplot(1, 1, 1)
        plt.imshow(np.rot90(im_cor_seg))
        a.axis('off')
        plt.savefig(img_seg, bbox_inches='tight', pad_inches=0)
        
        m.reset_state()
        
    np.savetxt(iou_file, np.array(result_metrica))
        
