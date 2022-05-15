# kmeans riemannian segmentation dt mri
 The objective of this project is to develop methodologies for filtering DT-MRI images and segmenting these images through the use of a new approach that consists of using the Riemannian metric as a target object in the construction of new clustering and filtering algorithms using mass centers produced in a Riemannian context, more precisely, in a non-Euclidean viewpoint.


# Usage
## Install dependences:
 - pip install -r requirements.txt

## For download and preprocessing datasets to run the script:
 - get_and_preprocessing_dataset.py

## For considering filtering process in the segmentation, run the script to generate the filtered dataset:
 - run_dti_riemannian_filter.py

## For execute the segmentation process:
 - run_segmentation.py

## For make results execute:
  - make_results.py
