#!/bin/bash

# Set job name
#SBATCH --job-name=slivit
# Specify the number of nodes and processors and gpus per nodes
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --cpus-per-task=17

#SBATCH --gres=gpu:1

# For ascend cluster, we have nextgen and quad nodes
#SBATCH --partition=nextgen


# Specify the amount of time for this job
#SBATCH --time=36:00:00

# Specify the maximum amount of physical memory required
#SBATCH --mem=64gb

# Specify an account when more than one available
#SBATCH --account=PCON0023


#SBATCH --output=results/%j_0_log.out

#SBATCH --error=results/%j_0_log.err


# Load modules:
module load cuda/11.8.0

module load miniconda3/24.1.2-py310

source activate slivit

cd /fs/ess/PCON0023/shileicao/code/SLIViT

python finetune.py \
       --dataset oct3d \
       --fe_path ./checkpoints/kermany/feature_extractor.pth \
       --out_dir ./results \
       --meta /fs/ess/PCON0023/eye3d/data/ukbiobank/oct \
       --task cls


