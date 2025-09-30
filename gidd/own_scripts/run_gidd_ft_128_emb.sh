#!/usr/bin/bash
torchrun --nnodes 1 --nproc_per_node 1 ../train.py --config-name gidd_ft_emb model.p_uniform=0.1 logging.run_name="'small-gidd-cond-emb-owt-pu=0.1-seq-len-128-start-128'" path='/home/vmeshchaninov/aiusupov/gidd-checkpoints/gidd-train-128' hydra.run.dir='/home/vmeshchaninov/aiusupov/gidd-outputs/2025-09-15/train_emb_from_128_seqlen' model.max_seq_len=128
