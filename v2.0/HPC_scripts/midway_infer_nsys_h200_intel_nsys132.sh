#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=01:00:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0602   # Intel Gold-6542Y, 1TB, H200 DLC
                                   # alternatives: midway3-0603/0604/0605/0606
#SBATCH -o midway_infer_nsys_nsys132_%N_%j.out
#SBATCH -e midway_infer_nsys_nsys132_%N_%j.err

# nsys-binary-swap diagnostic sibling of midway_infer_nsys_h200_intel.sh.
#
# Purpose:
#   Three independent Intel production runs (noforktrace 50084384,
#   newinfer 50084385, delay 50102626) produced byte-identical CUPTI
#   counts -- RUNTIME=4252 / MEMCPY=1960 / SYNC=1960, KERNEL/MEMSET/
#   CUDA_EVENT/OVERHEAD all missing -- across very different nsys flag
#   combinations and across both inference scripts. Even --delay=15
#   produced the same ~3 s capture window, just shifted later in the
#   run -- proving the failure is "CUPTI dies after ~3 s of capture
#   regardless of when capture starts," not "CUPTI dies at a specific
#   point in the workload."
#
#   That pattern is consistent with a bug in nsys 2025.1.3 (which ships
#   with cuda/12.9 on Midway) + driver 535.216.03 + sustained
#   high-kernel-rate Hopper workloads. The earlier kernel-diag matrix
#   confirmed cuda-13.2/bin/nsys (~2025.5+) is available on Midway and
#   captures kernels fine on the matmul probe.
#
# Test:
#   Invoke /software/cuda-13.2-el8-x86_64/bin/nsys with an absolute
#   path. Keep cuda/12.9 module loaded so the CUDA runtime the
#   workload links against stays unchanged -- the ONLY variable vs
#   the broken original production capture is the nsys binary.
#
#   - KERNEL captured -> nsys 2025.1.3 bug confirmed. Production fix:
#                        switch all profile scripts to call cuda-13.2/
#                        bin/nsys explicitly (one-line PATH/binary
#                        change).
#   - KERNEL missing  -> the bug isn't in the nsys binary alone;
#                        likely a deeper CUPTI / driver issue and we
#                        need RCC to update the driver or to file an
#                        nsys bug report.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

# Override only the nsys binary -- leave everything else the same.
NSYS=/software/cuda-13.2-el8-x86_64/bin/nsys
echo "nsys binary override: ${NSYS}"
echo "nsys version: $("${NSYS}" --version 2>&1 | head -1)"
echo "(for reference, default cuda/12.9 nsys would be: $(which nsys))"

export WANDB_MODE=offline

echo "=== midway_infer_nsys_nsys132 H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
nvidia-smi topo -m

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_nsys132_${SLURM_JOB_ID}"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

# Original production nsys flags intentionally preserved
# (--trace-fork-before-exec=true present, no --sample=none, no --delay,
# no --cuda-flush-interval) so the nsys binary swap is the only variable
# vs the broken pre-flush original capture.
"${NSYS}" profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="infer_nsys_h200_intel"

echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
