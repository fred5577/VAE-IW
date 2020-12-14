#!/bin/sh
cd ..

epochs=100
batch_size=64
zdim=200
beta=0.0001
image_training_size=128
temp=5
annealing=True
file_naming=Adam_1e-4
kernel_size=32
loss=MSE

code_to_run=""
out=""
error=""
name=""
myfunc()
{
    code_to_run="python3 gumbel_res_${kernel_size}x${kernel_size}.py --epochs $epochs --batch-size $batch_size --env $1 --zdim $zdim --sigma 1 --beta $beta --annealing $annealing --image-training-size $image_training_size --temp $temp --file-naming $file_naming --loss $loss"
    name="$1-sigma_1_beta_${beta}_${zdim}_${kernel_size}_${kernel_size}_${loss}_A"
    out="../data/${1}/output-model/model_sigma_1_beta_${beta}_${zdim}_${kernel_size}_${kernel_size}_${loss}_${file_naming}.out"
    err="../data/${1}/output-model/model_sigma_1_beta_${beta}_${zdim}_${kernel_size}_${kernel_size}_${loss}_${file_naming}.err"
}

myfunc Alien
echo $code_to_run
bsub -q gpuv100 -J $name -gpu "num=1:mode=exclusive_process" -n 1 -W 24:00 -R "rusage[mem=4GB]" -B -N -o $out -e $err $code_to_run

myfunc Amidar
echo $code_to_run
bsub -q gpuv100 -J $name -gpu "num=1:mode=exclusive_process" -n 1 -W 24:00 -R "rusage[mem=4GB]" -B -N -o $out -e $err $code_to_run

myfunc Asteroids
echo $code_to_run
bsub -q gpuv100 -J $name -gpu "num=1:mode=exclusive_process" -n 1 -W 24:00 -R "rusage[mem=4GB]" -B -N -o $out -e $err $code_to_run

myfunc BattleZone
echo $code_to_run
bsub -q gpuv100 -J $name -gpu "num=1:mode=exclusive_process" -n 1 -W 24:00 -R "rusage[mem=4GB]" -B -N -o $out -e $err $code_to_run

myfunc BeamRider
echo $code_to_run
bsub -q gpuv100 -J $name -gpu "num=1:mode=exclusive_process" -n 1 -W 24:00 -R "rusage[mem=4GB]" -B -N -o $out -e $err $code_to_run

myfunc Gopher
echo $code_to_run
bsub -q gpuv100 -J $name -gpu "num=1:mode=exclusive_process" -n 1 -W 24:00 -R "rusage[mem=4GB]" -B -N -o $out -e $err $code_to_run

