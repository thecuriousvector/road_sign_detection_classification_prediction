#!/bin/bash


#SBATCH --job-name=disp_net
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:p40:1
#SBATCH --mail-type=END
#SBATCH --mail-user=rdv253@nyu.edu



module purge
module load python3/intel/3.5.3
##module load jupyter-kernels/py3.5
source /scratch/rdv253/CV/Project/tensorflow-gpu/tensorflow/py3.5.3/bin/activate
module load cuda/8.0.44
module load cudnn/7.0v4.0
cd /scratch/rdv253/CV/Project/depth_prediction/
python main.py --dataset_dir=/scratch/rdv253/CV/Project/Data/Processed/ --checkpoint_dir=/scratch/rdv253/CV/Project/depth_prediction/checkpoints/ --img_width=416 --img_height=128 --batch_size=4


