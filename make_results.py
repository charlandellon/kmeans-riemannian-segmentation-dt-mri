import os
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
from tensorflow.keras.metrics import MeanIoU
from dipy.reconst.dti import decompose_tensor, color_fa, fractional_anisotropy


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

fmask = os.path.sep.join([inputpath, 'stanford_hardi_denoised_mask.nii.gz'])
mask, _ = load_nifti(fmask)
mask = mask.astype(np.bool8)

# dataset DTI image used in the segmentation 
input_file = 'stanford_hardi_denoised_dti.nii.gz'
dti_dir = os.path.sep.join([outputpath, 'dti', 'images'])
if not os.path.exists(dti_dir):
    os.makedirs(dti_dir)

fdti = os.path.sep.join([inputpath, input_file])
dti, affine = load_nifti(fdti)
evals, evecs = decompose_tensor(dti)
FA = fractional_anisotropy(evals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, evecs)
cfa = RGB
cfa /= cfa.max()

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
plt.show()

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

volume_segmented[idx_class_1] = 2
volume_segmented[idx_class_2] = 1
# volume_segmented[idx_class_3] = 1 # For 3 or 4 classes uncomment this line
# volume_segmented[idx_class_4] = 4 # For 4 classes uncomment this line

cor_seg[idx_class_1] = (1, 0, 0)
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
plt.show()


fatias_sagital = np.arange(0, class_referencia.shape[0], 5)
fatias_coronal = np.arange(0, class_referencia.shape[1], 5)
fatias_axial = np.arange(0, class_referencia.shape[2], 5)

fatias_dic = {'sagital':fatias_sagital, 'coronal':fatias_coronal, 'axial': fatias_axial}


m = MeanIoU(num_classes=n_classes + 1)

iou_file_total = os.path.sep.join([outputpath, 'segmented', 'data', 'iou_total.txt'])
m.update_state(class_referencia.flat, volume_segmented.flat)
np.savetxt(iou_file_total, np.array(m.result().numpy()).reshape((-1,1)))
        
m.reset_state()

from dipy.data import get_sphere
sphere = get_sphere('repulsion724')
from dipy.viz import window, actor
            
for k, fatias in fatias_dic.items():
    
    
    iou_file = os.path.sep.join([outputpath, 'segmented', 'data', 'iou_' + k +'.txt'])
    
    result_metrica = []

    for v in fatias:
        data_seg = os.path.sep.join([outputpath, 'segmented', 'data', f'segmented_DTI_3D_slice_{v}_'+ k +'.nii.gz'])
        img_seg = os.path.sep.join([outputpath, 'segmented', 'images', f'segmented_DTI_3D_slice_{v}_'+ k +'.png'])
        img_ref = os.path.sep.join([outputpath, 'segmented', 'images', f'referencia_DTI_3D_slice_{v}_'+ k +'.png'])
        img_dti = os.path.sep.join([dti_dir, f'DTI_3D_slice_{v}_'+ k +'.png'])
        
        scene = window.Scene()

        if k == 'sagital':
            image_ref = class_referencia[v,:,:]
            image_seg = volume_segmented [v,:,:]
            im_cor_ref = cor_ref[v, :, :, :]
            im_cor_seg = cor_seg[v, :, :, :]
            evals_fatia = np.moveaxis(evals[v:v+1, :, :], 0, 2)
            evecs_fatia = np.moveaxis(evecs[v:v+1, :, :], 0, 2)
            cfa_fatia = np.moveaxis(cfa[v:v+1, :, :], 0, 2)
            mask_f = np.moveaxis(mask[v:v+1, :, :], 0, 2)
            
            
        elif k == 'coronal':
            image_ref = class_referencia[:,v,:]
            image_seg = volume_segmented [:,v,:]
            im_cor_ref = cor_ref[:, v, :, :]
            im_cor_seg = cor_seg[:, v, :, :]

            evals_fatia = np.moveaxis(evals[:, v:v+1, :], 1, 2)
            evecs_fatia = np.moveaxis(evecs[:, v:v+1, :], 1, 2)
            cfa_fatia = np.moveaxis(cfa[:, v:v+1, :], 1, 2)
            mask_f = np.moveaxis(mask[:, v:v+1, :], 1, 2)
            
        elif k == 'axial':
            image_ref = class_referencia[:,:,v]
            image_seg = volume_segmented [:,:,v]
            im_cor_ref = cor_ref[:, :, v, :]
            im_cor_seg = cor_seg[:, :, v, :]
            evals_fatia = evals[:, :, v:v+1]
            evecs_fatia = evecs[:, :, v:v+1]
            cfa_fatia = cfa[:, :, v:v+1]
            mask_f = mask[:, :, v:v+1]

        # Criação do ator tensor slicer para a fatia escolhida
        tensor_actor = actor.tensor_slicer(evals_fatia, evecs_fatia, 
                                           scalar_colors=cfa_fatia, 
                                           sphere=sphere, scale=0.7, 
                                           opacity=1.0, mask=mask_f)
        
        if tensor_actor is not None:
            scene.add(tensor_actor)
            
            # Define a cor do fundo da cena (R, G, B)
            scene.background((0, 0, 0))  # Branco
        
            # Ajusta a câmera para visualizar a fatia de frente
            scene.reset_camera()
            # scene.set_camera(position=pos_camera, focal_point=focal_point, view_up=view_up)
            scene.zoom(1.5)  # Ajuste o fator de zoom conforme necessário
        
            print('Saving illustration as tensor_ellipsoids.png')
            window.record(scene, n_frames=1, out_path=img_dti, 
                          size=(800, 800), magnification=1)
        else:
            print("Erro: tensor_actor é None.")
    
        
                    
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


    



    
        
