#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Amidar-sigma_1_beta_0-0001_200_4_4_MSE_A
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB BSUB -B
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ../../data/Amidar/output-model/model_sigma_1_beta_0-0001_200_4_4_MSE_A.out
#BSUB -e ../../data/Amidar/output-model/model_sigma_1_beta_0-0001_200_4_4_MSE_A.err
# -- end of LSF options --
cd ..

epochs = $1
batch_size = $2
env = $3
zdim = $4
beta = $5
image_training_size = $6
temp = $7
annealing = $8
file_naming = $9
echo $epochs

nvidia-smi

python3 gumbel_res_4x4.py --epochs $epochs --batch-size $batch_size --env $env --zdim $zdim --sigma 1 --beta $beta --image-training-size $image_training_size --temp $temp --annealing $annealing --file-naming $file_naming
