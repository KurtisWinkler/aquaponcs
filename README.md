# aquaponcs
Algorithm to Quantify Properties of Nuclear and Chromatin Structure

## Installation
conda create -n aqua\
source activate aqua

python 3.10.4

conda install pip (v. 22.1.2)\
pip install numpy (v. 1.23.3)\
pip install pandas (v. 1.5.2)\
pip install matplotlib (v. 3.6.0)\
pip install opencv-python (v. 4.6.0.66)\
pip install -U scikit-image (v. 0.19.3)

## Purpose 
Visualizing chromatin dispersal and nuclear structure can reveal information about cell stability, gene expression, and cell fate. Current image processing pipelines are largely qualitative, creating a need for an algorithm to quantify nuclear structure and detect/measure properties of chromatin blobs. AQUAPONCS utilizes openCV and scikit image packages to process input images, identify nucleus contours, identify chromatin blob contours, measure parameters of blobs, and filter/score blobs. The software outputs a processed image along with a .csv file quantifying the structure of the nucleus and chromatin.  

## Usage
### Blob detection 
The main workflow for blob detection is carried out through blob_main.py. Users must define the following arguments as described in blob_args.py: 

--file_name (str): the name of the input image file to be processed
--min_distance (int): minimum pixel distance between blob maxima. Decreasing will allow for greater granularity for detection of overlapping blobs (default: 10)
--min_thresh_maxima (float): minimum relative intensity threshold for maxima (default: 0.8)
--min_thresh_contours (float): minimum relative threshold for contours (default: 0.8)
--thresh_step (int): step size for finding contours (default: 5)
--init_filter (list): the initial filter for removing blobs. This could be used for removing blobs based on user-defined parameters (ie, too small of area, non-uniform circularity). (default: None)
--sim_filter (list): filters for similar blobs. Differentiates between unique blobs (default: None)
--out_filter (list): filters for outlier blobs whose parameters fall outside of a user-defined range (default: None) 
--best_filter (list): user-defined criteria to determine which parameters are used to score and identify the best blobs (default: None) 

If the user does not wish to include initial, similar, or outlier filters, the following optional arguments can be added: 
--no_init_filter
--no_sim_filter
--no_out_filter

## Contrast Function
The contrast function rescales input images by a percentage of pixel intensity defined by the user. It takes arguments for:
file (str): the name of the input image
max_rescale (int): the pixel intensity value to rescale the current maximum up to
min_rescale (int): the pixel intensity value to rescale the current maximum down to
output_name (str): name of the proessed output image to be used in workflow downstream 

The output of the function is a numpy array of pixel intensities and an output file image. 

## example code 
