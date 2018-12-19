# Road Sign Detection

This module takes in the KITTI raw images (City, Residential and Road) and identifies the pictures with the road sign. The output includes a new image with detection box around the road sign classified into 3 categories - mandatory, prohibitory and danger - along with their probabilities. In addition to this, we've also generated CSVs for all the images with road sign that give us the coordinates of the detection box, category the image belongs to (City/Residential/Road), relative path of the image in KITTI dataset and classification category with the probablity. 

Note: We've used the pretrained model from "https://github.com/aarcosg/traffic-sign-detection" and modified it according to our needs. We've used RFCN Resnet 101 for detecting the road signs.

## Step 1
Replace the visualisation_utils.py file from the tensorflow_modules/ folder with the actual file in the python's site-packages folder. We've added the code for generating the coordinates for the boxes generated after detection.

## Step 2
Edit the folder path in the "Run_models_on_new_images_Ramya.ipynb" file to include the images you wish to perform detection on.

## Step 3
The slurm script for performing detection is provided in scripts/ folder. Modify according to your needs and run the sbatch.

The total time for prediction is 30.67 hours. But we parallelized the process by running multiple slurs jobs.
