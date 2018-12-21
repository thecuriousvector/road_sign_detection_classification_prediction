# Road Sign Depth Prediction

This module includes the code associated with the training and testing of the depth prediction model inspired from SFMLearner (https://github.com/tinghuiz/SfMLearner) and DispNet (https://arxiv.org/pdf/1512.02134.pdf). SfMLearner jointly trains single view depth estimation and mutliple view pose estimation models. But, we've separated the depth estimation model from pose estimation and employed it in our project.

The depth estimation module in SfMLearner uses the architecture of DispNet. The evaluation scripts used in SfMLearner are based on Godard et al's codebase (https://github.com/mrharicot/monodepth).


## Step 1: Preprocessing
Please reach out to the authors for pre-processed data

## Step 2: Training
python main.py --dataset_dir=<path to preprocessed data> --checkpoint_dir=<path to checkpoints folder> --img_width=416 --img_height=128 --batch_size=4
  
## Step 3: Testing
python evaluation.py --kitti_dir=<path to raw dataset> --pred_file=<path to .npy file>


