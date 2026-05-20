#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # smoke test only needs one GPU
#SBATCH --cpus-per-task=4
#SBATCH -o smoke_d2h_nvidia_%j.out
#SBATCH -e smoke_d2h_nvidia_%j.err

# Smoke test: isolates .item() vs bulk numpy and sync vs async D2H transfer
# patterns from inference.py vs inference_optimized.py.
# Does NOT require a model checkpoint — pure PyTorch tensor operations only.

ulimit -l unlimited

module load apptainer

export NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname)   GPUs available: ${NUM_GPUS}   Job: ${SLURM_JOB_ID}"
nvidia-smi

export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD='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'

apptainer exec \
    --nv \
    --bind /lustre/fs01 \
    /home/ucg-aepmn/uchigaco/pytorch_25.10.sif \
    bash -c "
        pip install ruamel.yaml -q --user 2>/dev/null;
        PYTHONPATH=/home/ucg-aepmn/uchigaco/S2S/v2.0 \
        python /home/ucg-aepmn/uchigaco/S2S/v2.0/test/d2h_pattern_smoke.py
    "
