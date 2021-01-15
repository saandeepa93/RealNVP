#!/bin/bash
source activate /data/saandeepaath/my_envs/.flow2
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /data/saandeepaath/flow_based_2/train.py --no_cuda False --root /data/saandeepaath/flow_based_2/
conda deactivate