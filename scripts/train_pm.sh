#!/bin/bash
#SBATCH --job-name=train_segformer3d_pm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=20:00:00
#SBATCH --output=/gpfs/home6/pgort1/projects/SegFormer3D/scripts/slurm_logs/%x_%j.out

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
export PM_PREPROCESSED_ROOT="/gpfs/work2/0/prjs1518/projects/SegFormer3D/Dataset101_PM_preprocessed_2mm"
export PM_PREPROCESS_SCRIPT="${SEGFORMER3D_HOME}/data/dataset101_pm/dataset101_pm_raw_data/dataset101_pm_preprocess.py"
export PM_SPLIT_SCRIPT="${SEGFORMER3D_HOME}/data/dataset101_pm/dataset101_pm_raw_data/datameta_generator/create_train_val_test_csv.py"
export EXPERIMENT_DIR="${SEGFORMER3D_HOME}/experiments/dataset101_pm/default_experiment"
export JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
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
"${ENV_PYTHON}" -c "import sys, nibabel; print('[info] exe:', sys.executable); print('[info] nibabel:', nibabel.__version__)"

mkdir -p "${PM_PREPROCESSED_ROOT}"

if [[ -z "$(ls -A "${PM_PREPROCESSED_ROOT}" 2>/dev/null)" ]] || [[ ! -f "${PM_PREPROCESSED_ROOT}/train.csv" ]] || [[ ! -f "${PM_PREPROCESSED_ROOT}/validation.csv" ]]; then
  echo "[info] preprocessing Dataset101_PM into ${PM_PREPROCESSED_ROOT}"
  "${ENV_PYTHON}" "${PM_PREPROCESS_SCRIPT}" --dataset-root "${PM_RAW_DATASET_ROOT}" --save-dir "${PM_PREPROCESSED_ROOT}"
fi

if [[ ! -f "${PM_PREPROCESSED_ROOT}/train.csv" ]] || [[ ! -f "${PM_PREPROCESSED_ROOT}/validation.csv" ]]; then
  echo "[info] generating train/validation CSV splits in ${PM_PREPROCESSED_ROOT}"
  "${ENV_PYTHON}" "${PM_SPLIT_SCRIPT}" --folder-dir "${PM_PREPROCESSED_ROOT}" --save-dir "${PM_PREPROCESSED_ROOT}" --append-dir "${PM_PREPROCESSED_ROOT}"
fi

"${ENV_PYTHON}" - <<'PY'
import os
import yaml

experiment_dir = os.environ["EXPERIMENT_DIR"]
job_config_path = os.environ["JOB_CONFIG_PATH"]
job_tag = os.environ["JOB_TAG"]

source_config_path = os.path.join(experiment_dir, "config.yaml")
with open(source_config_path, "r", encoding="utf-8") as infile:
    config = yaml.safe_load(infile)

config["training_parameters"]["checkpoint_save_dir"] = f"model_checkpoints/best_dice_checkpoint_{job_tag}"

with open(job_config_path, "w", encoding="utf-8") as outfile:
    yaml.safe_dump(config, outfile, sort_keys=False)

print(f"[info] checkpoint dir: {config['training_parameters']['checkpoint_save_dir']}")
print(f"[info] config: {job_config_path}")
PY

cd "${EXPERIMENT_DIR}"
"${ENV_ACCELERATE}" launch --config_file "./gpu_accelerate.yaml" run_experiment.py --config "${JOB_CONFIG_PATH}"

