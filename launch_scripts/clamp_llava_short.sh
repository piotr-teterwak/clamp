#!/bin/bash -x

#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=llava_clamp_short
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
srun --cpu_bind=v --accel-bind=gn python -u  src/training/main.py --train-data '/fsx/youngkyun/data/cc3m/cc3m/{00000..00331}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/0/{00000..01066}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/1/{00000..01067}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/2/{00000..01014}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/3/{00000..01085}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/4/{00000..01049}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/5/{00000..01067}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/6/{00000..01063}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/7/{00000..01038}.tar::/fsx/pteterwak/data/cc12m/cc12m/{000000..001240}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/8/{00000..01063}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/9/{00000..01000}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/10/{00000..01017}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/11/{00000..01010}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/12/{00000..01021}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/13/{00000..01039}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/14/{00000..01032}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/15/{00000..01026}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/16/{00000..01054}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/17/{00000..01069}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/18/{00000..01051}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/19/{00000..01031}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/20/{00000..01061}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/21/{00000..01072}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/22/{00000..01049}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/23/{00000..01054}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/24/{00000..01046}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/25/{00000..01023}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/26/{00000..01049}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/27/{00000..01055}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/28/{00000..01050}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/29/{00000..01015}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/30/{00000..01033}.tar::/fsx/youngkyun/data/cc12m/im21k_wrapped/data/{000000..000332}.tar::/fsx/youngkyun/data/cc12m/{00000..01242}.tar'  --train-num-samples 10000000 --dataset-type webdataset_double_tokenizer   --batch-size 64 --zeroshot-frequency 1  --precision amp --workers 5  --dataset-resampled --model clamp_llava  --gather-with-grad --epochs 6 --lr 0.0005 --wd 0.5 --warmup 1220 --eps 1e-08 --local-loss --lock-image   --pretrained openai   --wrap-caption-long-list --logs /fsx/youngkyun/clamp/logs/ --grad-checkpointing --grad-clip-norm 1.0 --distill-model ViT-L-14-336 --distill-pretrained openai


