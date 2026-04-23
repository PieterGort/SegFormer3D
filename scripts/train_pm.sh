#!/bin/bash
#SBATCH --job-name=train_segformer3d_pm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:30:00
#SBATCH --output=/gpfs/home6/pgort1/projects/SegFormer3D/scripts/slurm_logs/%x_%j.out

# Launcher for Dataset101_PM training.
#
#   • Submit 5-fold CV as a SLURM array job:
#         sbatch --array=1-5 train_pm.sh
#     Each array task trains one fold on its own GPU/time budget.
#
#   • Single-fold run (array index unset): defaults to fold 1, or set PM_FOLD.
#         PM_FOLD=3 sbatch train_pm.sh
#
# Preprocessing and fold-CSV generation happen once (guarded on file presence).
# Concurrent array tasks may race on the first run; re-submit if that happens.

set -euo pipefail

mkdir -p /gpfs/home6/pgort1/projects/SegFormer3D/scripts/slurm_logs

module purge
export SEGFORMER3D_ENV="/gpfs/home6/pgort1/.conda/envs/segformer3d"
export ENV_PYTHON="${SEGFORMER3D_ENV}/bin/python"
export ENV_ACCELERATE="${SEGFORMER3D_ENV}/bin/accelerate"

export WANDB_MODE="online"
unset WANDB_API_KEY
export SEGFORMER3D_HOME="/gpfs/home6/pgort1/projects/SegFormer3D"
export PM_RAW_DATASET_ROOT="/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM"
# v3 preprocessing: same 2 mm isotropic spacing + center-crop-or-pad as v2,
# but uses nnU-Net CTNormalization (dataset-global percentile clip + z-score)
# instead of v2's per-case z-score + hard-coded [-175, 250] HU clip. This is
# the "step B" change in our A/B/C preprocessing roadmap.
export PM_PREPROCESSED_ROOT="/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed_2mm_v3_ctnorm"
export PM_PREPROCESS_SCRIPT="${SEGFORMER3D_HOME}/data/dataset101_pm/dataset101_pm_raw_data/dataset101_pm_preprocess.py"
export PM_KFOLD_SCRIPT="${SEGFORMER3D_HOME}/data/dataset101_pm/dataset101_pm_raw_data/datameta_generator/create_train_val_kfold_csv.py"
export EXPERIMENT_DIR="${SEGFORMER3D_HOME}/experiments/dataset101_pm/default_experiment"

# Select fold: SLURM_ARRAY_TASK_ID > PM_FOLD > 1 (default).
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  export FOLD="${SLURM_ARRAY_TASK_ID}"
elif [[ -n "${PM_FOLD:-}" ]]; then
  export FOLD="${PM_FOLD}"
else
  export FOLD="1"
fi

if ! [[ "${FOLD}" =~ ^[1-5]$ ]]; then
  echo "[error] FOLD must be 1..5, got: ${FOLD}" >&2
  exit 1
fi

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
JOB_ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
export JOB_TAG="${JOB_ARRAY_ID}_fold${FOLD}"
export JOB_CONFIG_PATH="${EXPERIMENT_DIR}/config_${JOB_TAG}.yaml"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "[error] Python not found in env: ${ENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${ENV_ACCELERATE}" ]]; then
  echo "[error] Accelerate not found in env: ${ENV_ACCELERATE}" >&2
  exit 1
fi

echo "[info] python: ${ENV_PYTHON}"
echo "[info] fold: ${FOLD}"
echo "[info] job tag: ${JOB_TAG}"
echo "[info] preprocessed root (v3 ctnorm): ${PM_PREPROCESSED_ROOT}"
"${ENV_PYTHON}" -c "import sys, nibabel; print('[info] exe:', sys.executable); print('[info] nibabel:', nibabel.__version__)"

mkdir -p "${PM_PREPROCESSED_ROOT}"

# Preprocess once: v3 runs a foreground-stats pass first, then preprocesses.
# meta.json is written after both passes complete, so its presence implies a
# full v3 output directory (otherwise we re-run preprocessing from scratch).
if [[ ! -f "${PM_PREPROCESSED_ROOT}/meta.json" ]]; then
  echo "[info] preprocessing Dataset101_PM (v3 ctnorm) into ${PM_PREPROCESSED_ROOT}"
  "${ENV_PYTHON}" "${PM_PREPROCESS_SCRIPT}" \
    --dataset-root "${PM_RAW_DATASET_ROOT}" \
    --save-dir "${PM_PREPROCESSED_ROOT}"
fi

# Generate 5-fold CSVs once (train_fold_{1..5}.csv + validation_fold_{1..5}.csv).
if [[ ! -f "${PM_PREPROCESSED_ROOT}/train_fold_1.csv" ]]; then
  echo "[info] generating 5-fold CSV splits in ${PM_PREPROCESSED_ROOT}"
  "${ENV_PYTHON}" "${PM_KFOLD_SCRIPT}" \
    --folder-dir "${PM_PREPROCESSED_ROOT}" \
    --save-dir "${PM_PREPROCESSED_ROOT}" \
    --append-dir "${PM_PREPROCESSED_ROOT}"
fi

# Render per-fold config: stamp the selected fold into both train/val dataset
# args and give this run a unique checkpoint dir + W&B name.
"${ENV_PYTHON}" - <<'PY'
import os
import yaml

experiment_dir = os.environ["EXPERIMENT_DIR"]
fold = int(os.environ["FOLD"])
job_tag = os.environ["JOB_TAG"]
job_config_path = os.environ["JOB_CONFIG_PATH"]

source_config_path = os.path.join(experiment_dir, "config.yaml")
with open(source_config_path, "r", encoding="utf-8") as infile:
    config = yaml.safe_load(infile)

config["dataset_parameters"]["train_dataset_args"]["fold_id"] = fold
config["dataset_parameters"]["val_dataset_args"]["fold_id"] = fold
config["training_parameters"]["checkpoint_save_dir"] = (
    f"model_checkpoints/best_dice_checkpoint_{job_tag}"
)
config["wandb_parameters"]["name"] = f"segformer3d_dataset101_pm_fold{fold}"

with open(job_config_path, "w", encoding="utf-8") as outfile:
    yaml.safe_dump(config, outfile, sort_keys=False)

print(f"[info] fold: {fold}")
print(f"[info] checkpoint dir: {config['training_parameters']['checkpoint_save_dir']}")
print(f"[info] wandb name: {config['wandb_parameters']['name']}")
print(f"[info] config: {job_config_path}")
PY

cd "${EXPERIMENT_DIR}"
"${ENV_ACCELERATE}" launch --config_file "./gpu_accelerate.yaml" run_experiment.py --config "${JOB_CONFIG_PATH}"
