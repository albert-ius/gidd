#!/usr/bin/bash
torchrun --nnodes 1 --nproc_per_node 1 ../train.py --config-name gidd_ft_128 model.p_uniform=0.1 logging.run_name="'small-gidd-owt-pu=0.1-seq-len-128'" path='/home/vmeshchaninov/aiusupov/gidd-checkpoints/gidd-small-pu-0.1' hydra.run.dir='/home/vmeshchaninov/aiusupov/gidd-outputs/2025-09-10/train_128' model.max_seq_len=128
