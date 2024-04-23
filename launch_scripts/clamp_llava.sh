#!/bin/bash -x

#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llava_clamp
#SBATCH --partition midas
#SBATCH -o /fsx/youngkyun/clamp/logs/slurm/%x_%A.o
#SBATCH -e /fsx/youngkyun/clamp/logs/slurm/%x_%A.e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=57129

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
#export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1


cd /data/home/youngkyun/piotr2/clamp
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -u  src/training/main.py --train-data '/fsx/youngkyun/data/cc3m/cc3m/{00000..00331}.tar'  --train-num-samples 10000000 --dataset-type webdataset_double_tokenizer   --batch-size 64 --zeroshot-frequency 1  --precision amp --workers 5  --dataset-resampled --model clamp_llava  --gather-with-grad --epochs 6 --lr 0.0005 --wd 0.5 --warmup 1220 --eps 1e-08 --local-loss --lock-image   --pretrained openai   --wrap-caption-long-list --logs /fsx/youngkyun/clamp/logs/ --grad-checkpointing --grad-clip-norm 1.0 --distill-model ViT-L-14 --distill-pretrained datacomp_xl_s13b_b90k

