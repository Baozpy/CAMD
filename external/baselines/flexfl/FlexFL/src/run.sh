# Agent4SR in the space reduction stage
CUDA_VISIBLE_DEVICES=? torchrun --nproc_per_node 1  --master_port=? pipeline.py
CUDA_VISIBLE_DEVICES=? torchrun --nproc_per_node 1  --master_port=? pipeline.py --dataset GHRB

# Combine results of SBIR, Ochiai, BoostN, Agent4SR
python combine.py
python combine.py --dataset GHRB

# Agent4LR in the localization refinement stage
CUDA_VISIBLE_DEVICES=? torchrun --nproc_per_node 1  --master_port=? pipeline.py --stage LR
CUDA_VISIBLE_DEVICES=? torchrun --nproc_per_node 1  --master_port=? pipeline.py --dataset GHRB --stage LR