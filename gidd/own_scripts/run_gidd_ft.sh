#!/usr/bin/bash
torchrun --nnodes 1 --nproc_per_node 1 ../train.py --config-name gidd_ft model.p_uniform=0.1 logging.run_name="'small-gidd-cond-emb-owt-pu=0.1'" path='/home/vmeshchaninov/aiusupov/gidd-checkpoints/gidd-small-pu-0.1' hydra.run.dir='/home/vmeshchaninov/aiusupov/gidd-outputs'
