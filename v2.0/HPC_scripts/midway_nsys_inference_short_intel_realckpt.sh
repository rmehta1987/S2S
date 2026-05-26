#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:20:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-[0602-0603]   # either Intel H200 test-partition node;
                                          # SLURM picks whichever is idle. Both are
                                          # Gold-6542Y / H200 DLC; the working short
                                          # capture (50072348) ran on 0603, the broken
                                          # production captures on 0602. Hardware spec
                                          # is identical, so cross-node scheduling
                                          # shouldn't introduce a confound.
#SBATCH -o midway_nsys_inference_short_intel_realckpt_%N_%j.out
#SBATCH -e midway_nsys_inference_short_intel_realckpt_%N_%j.err

# Purpose:
#   The working Midway inference capture (job 50072348, produced by
#   midway_nsys_inference_short_intel.sh) is the only Midway nsys profile
#   to date that successfully populated CUPTI_ACTIVITY_KIND_KERNEL
#   (158,339 events, full bf16/cuDNN/Hopper SM90 kernel mix). Every other
#   Midway inference capture -- noforktrace 50084384, newinfer 50084385,
#   delay 50102626, nsys132 50117688, noforktrace_noworkers 50134752, plus
#   the two pre-flush originals and the AMD/H100 variants -- shows the
#   byte-identical broken fingerprint RUNTIME=4252 / MEMCPY=1960 /
#   SYNC=1960 with KERNEL/MEMSET/CUDA_EVENT/OVERHEAD absent.
#
#   Comparing the working short script to the broken production scripts,
#   only two non-cosmetic axes differ:
#     (a) --duration=180   -- short has it, production scripts don't.
#     (b) checkpoint       -- short synthesises a stub
#                             ({'iters':0,'epoch':0,'model_state':{}}),
#                             production scripts torch.load() the real
#                             ~hundreds-of-MB checkpoint at
#                             results/S2S/infer_nsys_h200_intel/training_checkpoints/ckpt.tar.
#   Other potential variables (entry point, AMP, cuDNN, multi-rank, fork
#   tracing, nsys binary, num_data_workers) have all been falsified by
#   earlier diagnostics -- see memory: project_midway_cupti_kernel_missing.md
#   and the Test A verdict on job 50134752.
#
# Test:
#   This script holds --duration=180, the node, the nsys flags, and the
#   workload entry point identical to the working short capture, and
#   varies ONLY the checkpoint: point RUN_NUM at the production
#   "infer_nsys_h200_intel" directory so restore_checkpoint() loads the
#   real ckpt.tar that the broken production scripts also load.
#
#   Note on num_data_workers: this script intentionally does NOT apply
#   the num_data_workers=0 override that the other Intel test-partition
#   scripts now carry. The working short capture used num_workers=8
#   (default from exp2.yaml, before the Test A override existed), and
#   Test A already showed num_workers=0 does not rescue the broken
#   production captures. Holding num_workers=8 here means the ONLY
#   variable vs the working short_50072348 capture is real-vs-stub
#   checkpoint.
#
# Reading the SUMMARY at the end:
#   KERNEL captured (N events)
#       -> real-checkpoint load is NOT the trigger. The production-vs-short
#          difference is then most likely --duration=180 itself: when nsys
#          bounds the capture and SIGTERMs the workload, CUPTI's kernel
#          subscription survives; when the workload runs unbounded to
#          natural completion, CUPTI silently drops kernel data.
#          Production fix: add --duration=<window> to every nsys profile
#          call (one-line change to each midway_infer_nsys_h200_* script).
#
#   KERNEL table missing
#       -> real-checkpoint load IS the trigger. Some specific operation
#          inside restore_checkpoint() (torch.load deserialisation, the
#          subsequent .to(device) cascade of weight tensors, or an
#          optimizer-state restore that touches CUDA streams) poisons
#          CUPTI's kernel callback for the rest of the run. Next step is
#          a torch.utils.bottleneck-style bisection of restore_checkpoint.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

export WANDB_MODE=offline

echo "=== midway_nsys_inference_short_realckpt Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "nsys=$(which nsys)  $(nsys --version | head -1)"
nvidia-smi -L
echo

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
# Intentionally NOT applying the num_data_workers=0 override that the
# other Intel test-partition scripts now carry. The working short capture
# used num_workers=8 and Test A (job 50134752) already falsified
# num_workers=0 as a rescue. Holding num_workers=8 keeps real-vs-stub
# checkpoint as the ONLY variable in this run.

# Use the SAME RUN_NUM as the broken production midway_infer_nsys_h200_intel.sh
# so restore_checkpoint() picks up the real ckpt.tar.
RUN_NUM="infer_nsys_h200_intel"

NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_short_realckpt_${SLURM_JOB_ID}"

REAL_CKPT="${SLURM_SUBMIT_DIR}/results/S2S/${RUN_NUM}/training_checkpoints/ckpt.tar"
if [[ ! -s "${REAL_CKPT}" ]]; then
    echo "ERROR: real checkpoint not found at ${REAL_CKPT}" >&2
    echo "       The production midway_infer_nsys_h200_intel.sh assumes it" >&2
    echo "       exists at that path. Either restore it from where you got" >&2
    echo "       the broken production captures' checkpoint, or run" >&2
    echo "       midway_nsys_inference_short_intel.sh (which synthesises a" >&2
    echo "       stub) instead." >&2
    exit 1
fi
echo "Real checkpoint present: ${REAL_CKPT}"
ls -lh "${REAL_CKPT}"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

# Same trace flags as midway_nsys_inference_short_intel.sh:
# --duration=180 bounds the capture; nsys SIGTERMs the workload at that
# point so the job finalises in ~3 minutes regardless of how long the full
# inference loop would otherwise take.
nsys profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    --duration=180 \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="${RUN_NUM}"

echo "Profile written: ${NSYS_OUT}.nsys-rep"

if [[ ! -s "${NSYS_OUT}.nsys-rep" ]]; then
    echo "ERROR: ${NSYS_OUT}.nsys-rep is missing or empty; nsys did not finalize." >&2
    exit 1
fi

nsys export --type=sqlite -o "${NSYS_OUT}.sqlite" --force-overwrite=true "${NSYS_OUT}.nsys-rep"

if [[ ! -s "${NSYS_OUT}.sqlite" ]]; then
    echo "ERROR: sqlite export failed; ${NSYS_OUT}.sqlite is missing or empty." >&2
    exit 1
fi

echo
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "CUPTI activity tables present in ${NSYS_OUT}.sqlite:"
sqlite3 "${NSYS_OUT}.sqlite" "SELECT name FROM sqlite_master WHERE name LIKE 'CUPTI_ACTIVITY%' ORDER BY name;"
echo

for tbl in CUPTI_ACTIVITY_KIND_RUNTIME \
           CUPTI_ACTIVITY_KIND_MEMCPY \
           CUPTI_ACTIVITY_KIND_SYNCHRONIZATION \
           CUPTI_ACTIVITY_KIND_KERNEL \
           CUPTI_ACTIVITY_KIND_MEMSET \
           CUPTI_ACTIVITY_KIND_CUDA_EVENT \
           CUPTI_ACTIVITY_KIND_OVERHEAD; do
    n=$(sqlite3 "${NSYS_OUT}.sqlite" "SELECT count(*) FROM ${tbl};" 2>/dev/null)
    if [[ -z "$n" ]]; then
        printf "  %-40s TABLE MISSING\n" "$tbl"
    else
        printf "  %-40s %s events\n" "$tbl" "$n"
    fi
done

echo
KERNEL_COUNT=$(sqlite3 "${NSYS_OUT}.sqlite" "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null)
if [[ -z "$KERNEL_COUNT" ]]; then
    echo "VERDICT: KERNEL table missing -> real-checkpoint load IS the trigger."
    echo "         --duration=180 alone is NOT enough to rescue capture when the"
    echo "         workload also restores a real production checkpoint. Next step:"
    echo "         bisect restore_checkpoint() (torch.load vs load_state_dict vs"
    echo "         optimizer-state restore) to isolate which operation poisons CUPTI."
elif [[ "$KERNEL_COUNT" -eq 0 ]]; then
    echo "VERDICT: KERNEL table present but empty -> CUPTI subscribed but dropped"
    echo "         every kernel record. Unusual state; inspect nsys stderr above."
else
    echo "VERDICT: KERNEL captured (${KERNEL_COUNT} events) -> real-checkpoint load"
    echo "         is NOT the trigger. The discriminator vs the broken production"
    echo "         captures is then --duration=180 itself. Production fix: add"
    echo "         --duration=<reasonable_window> to every nsys profile call in"
    echo "         the midway_infer_nsys_h200_*.sh scripts."
fi
