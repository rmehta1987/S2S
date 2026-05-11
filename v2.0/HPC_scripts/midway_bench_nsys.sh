#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:45:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o midway_bench_nsys_%x_%j.out
#SBATCH -e midway_bench_nsys_%x_%j.err

# Nsight Systems profile of the benchmark window only.
# --capture-range=cudaProfilerApi means nsys records nothing until
# train.py calls cudaProfilerStart() at the first measured step
# (after warmup), and stops recording when it calls cudaProfilerStop()
# at the end of the last measured step. Warmup, NCCL init, and the
# _bench_finalize all_reduce are excluded from the trace.
#
# Output: ${SLURM_SUBMIT_DIR}/nsys_bench_<run_num>.nsys-rep
# To analyse on Midway after the job:
#   nsys stats nsys_bench_<run_num>.nsys-rep
#   nsys export --type=sqlite nsys_bench_<run_num>.nsys-rep
#   # then scp the .sqlite to your laptop and open with nsys-ui,
#   # or query it with sqlite3 (see bench_methodology.md for queries).

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Bench knobs — same as midway_bench.sh so results are comparable.
export S2S_BENCH=1
export S2S_BENCH_WARMUP=20
export S2S_BENCH_STEPS=80
export S2S_NVTX=1          # activate NVTX ranges + cudaProfilerStart/Stop

export S2S_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_results.csv"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
run_num="nsys_bench_$(date +%s)"

export S2S_YAML="$(realpath ${config_file})"
export S2S_RUN_NUM="${run_num}"

echo "=== midway_bench_nsys: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}  run_num=${run_num}"

NSYS_OUT="${SLURM_SUBMIT_DIR}/nsys_bench_${run_num}"
echo "nsys output: ${NSYS_OUT}.nsys-rep"

nsys profile \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${NSYS_OUT}" \
    --force-overwrite=true \
    --nic-metrics=true \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/train.py \
        --yaml_config="${config_file}" \
        --run_num="${run_num}"

echo "=== nsys export to sqlite ==="
nsys export --type=sqlite --output="${NSYS_OUT}.sqlite" "${NSYS_OUT}.nsys-rep"
echo "sqlite written: ${NSYS_OUT}.sqlite"
echo "scp that file here and run: sqlite3 <file> (see bench_methodology.md for queries)"
