#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --time=23:00:00
#SBATCH --mem=110G
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
## run the application:
job_name="test" # Name of the experiment
cfg_path="train_configs/224_v2_llama2_video_stage_3.yaml" # path to the config file
number_of_gpus=1 # number of gpus
# cd ../../

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done
echo "Port is $PORT"
 CUDA_VISIBLE_DEVICES=3,4 --nproc-per-node=2 
torchrun --master-port ${PORT} --nproc-per-node $number_of_gpus train.py --job_name ${job_name} --cfg-path ${cfg_path}
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=4,5 torchrun --master-port 2299 --nproc-per-node 2 train.py --job_name "test" --cfg-path "train_configs/224_v2_llama2_video_stage_3.yaml"  