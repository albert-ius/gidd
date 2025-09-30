#!/bin/bash
#SBATCH -A proj_1625
#SBATCH --constraint='type_e'
#SBATCH --error=ft_128_emb-%j.err        # Файл для вывода ошибок 
#SBATCH --output=ft_128_emb-%j.log       # Файл для вывода результатов 
#SBATCH --reservation=aicenter
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00

conda deactivate
conda activate gidd-env

echo "Working on node `hostname`" 
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"


cd /home/vmeshchaninov/aiusupov/gidd-fork/gidd/own_scripts
export WANDB_API=17fd83b074d2cc035f5a3037df5d59d57388a2b7
export WANDB_MODE=offline
./run_gidd_ft_128_emb.sh
