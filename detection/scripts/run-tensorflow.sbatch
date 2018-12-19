#!/bin/bash

##SBATCH --job-name=jupyterTest2
##SBATCH --nodes=1
##SBATCH --cpus-per-task=2
##SBATCH --mem=2GB
##SBATCH --time=120:00:00

#SBATCH --job-name=lang_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:p40:1
#SBATCH --mail-type=END
#SBATCH --mail-user=rdv253@nyu.edu



module purge
#module load jupyter-kernels/py2.7
##module load jupyter-kernels/py3.5
##module load tensorflow/python3.5/1.4.0
module load python3/intel/3.5.3
module load jupyter-kernels/py3.5
source /scratch/rdv253/CV/Project/tensorflow-gpu/tensorflow/py3.5.3/bin/activate

##cd /scratch/rdv253/CV/Project/cocapi/PythonAPI
##make
##cp -r pycocotools /home/wang/tensorflow/models/research/
##cd /home/wang/tensorflow/models/research/
##protoc object_detection/protos/*.proto --python_out=.
cd /scratch/rdv253/CV/Project/tensorflow-gpu/tensorflow/py3.5.3/lib/python3.5/site-packages/tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd /scratch/rdv253/CV/Project/traffic-sign-detection-master

port=$(shuf -i 6000-9999 -n 1)


/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to prince with command

ssh $USER@prince

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port




