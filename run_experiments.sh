#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_experiments.sh (based on hip_research.main.model_eval)
# - Supports jobs: ppl, passkey, stream
# - This script is intentionally verbose for reproducibility
# ============================================================

# ------------------------------------------------------------
# HIP_EXTEND:
# - Controls HiP self-extend (RoPE extension related)
# - Default: 0 (disabled)
# ------------------------------------------------------------
export HIP_EXTEND="${HIP_EXTEND:-0}"

# ------------------------------------------------------------
# Model / method configuration
# ------------------------------------------------------------
MODEL="${MODEL:-llama3.1_8b_instruct}"

# METHODS:
# - Space-separated list
# - Executed in order
# - Example: "fa2 hip adahip"
METHODS="${METHODS:-fa2 hip adahip}"

# JOBS:
# - ppl, passkey, stream
JOBS="${JOBS:-ppl}"

# ------------------------------------------------------------
# PPL experiment configuration
# ------------------------------------------------------------
# DATASETS_PPL:
# - wikitext / pg19
DATASETS_PPL=(${DATASETS_PPL:-wikitext pg19})

# STRIDES:
# - Context window stride for evaluation
# 4096, 8192, 16384, 32768, 65536, 131072
STRIDES=(${STRIDES:-4096 8192 16384 32768 65536 131072})

# KS:
# - Attention budget (meaningful for hip/adahip)
# 64, 128, 256, 512, 1024, 2048
KS=(${KS:-256 512 1024 2048})

# COUNT:
# - Number of evaluation segments
# - -1 means "use full available segments"
COUNT="${COUNT:--1}"

# BATCH_SIZE:
# - Usually kept at 1 for long-context PPL stability
BATCH_SIZE="${BATCH_SIZE:-1}"

# ------------------------------------------------------------
# HiP / AdaHiP block configuration
# ------------------------------------------------------------
BLOCK_SIZE_Q="${BLOCK_SIZE_Q:-32}"
BLOCK_STRIDE_Q="${BLOCK_STRIDE_Q:-32}"
BLOCK_SIZE_K="${BLOCK_SIZE_K:-2}"
BLOCK_STRIDE_K="${BLOCK_STRIDE_K:-1}"

# Number of dense (full attention) layers at bottom
DENSE_LAYERS="${DENSE_LAYERS:-3}"

# ------------------------------------------------------------
# AdaHiP sweep configuration
# ------------------------------------------------------------
# ALPHAS:
# - Entropy sensitivity coefficient
# POWERS:
# - Nonlinear shaping of entropy -> k mapping
ALPHAS_STR="${ALPHAS:-0.75 1.0 1.25}"
POWERS_STR="${POWERS:-0.5 1.0 2.0}"

read -r -a ALPHAS_ARR <<< "${ALPHAS_STR}"
read -r -a POWERS_ARR <<< "${POWERS_STR}"

# ------------------------------------------------------------
# Result directories
# ------------------------------------------------------------
ROOT_DIR="${ROOT_DIR:-/work}"
RESULTS_ROOT="${ROOT_DIR}/results"

# [MOD] Separate directories for logs and summaries
LOG_DIR="${RESULTS_ROOT}/logs"
SUMMARY_DIR="${RESULTS_ROOT}/summaries"

mkdir -p "${LOG_DIR}" "${SUMMARY_DIR}"

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
ts() { date +"%Y%m%d_%H%M%S"; }

have_job() {
  [[ " ${JOBS} " == *" $1 "* ]]
}

is_fa2() { [[ "$1" == "fa2" ]]; }
is_adahip() { [[ "$1" == "adahip" ]]; }

# Convert float to filename-safe token (e.g., 1.0 -> 1p0)
fmt_num() { echo "${1//./p}"; }

# ------------------------------------------------------------
# Common arguments shared across all jobs
# ------------------------------------------------------------
common_args() {
  local method="$1"
  echo \
    --model "${MODEL}" \
    --method "${method}" \
    --batch_size "${BATCH_SIZE}" \
    --overwrite \
    --block_size_q "${BLOCK_SIZE_Q}" \
    --block_stride_q "${BLOCK_STRIDE_Q}" \
    --block_size_k "${BLOCK_SIZE_K}" \
    --block_stride_k "${BLOCK_STRIDE_K}" \
    --dense_layers "${DENSE_LAYERS}"
}

# ------------------------------------------------------------
# Summary generation
# ------------------------------------------------------------
# [MOD]
# - Extract only per-step PPL lines
# - Append DONE marker at the end
make_summary() {
  local logfile="$1"
  local summaryfile="$2"
  local tag="$3"

  {
    tr '\r' '\n' < "${logfile}" \
      | grep -oE 'step[[:space:]]+[0-9]+[[:space:]]+PPL:[^[:cntrl:]]*sec' \
      || true
    echo "------------------------------------------------------------"
    echo "[${tag}] DONE $(date -Is)"
  } > "${summaryfile}"
}

# ------------------------------------------------------------
# Logging wrappers
# ------------------------------------------------------------
run_and_log() {
  local tag="$1"; shift
  local logfile="${LOG_DIR}/${tag}.log"
  local summaryfile="${SUMMARY_DIR}/${tag}.summary.txt"

  echo "============================================================" | tee "${logfile}"
  echo "[${tag}] $(date -Is)" | tee -a "${logfile}"
  echo "CMD: $*" | tee -a "${logfile}"
  echo "------------------------------------------------------------" | tee -a "${logfile}"

  ( set -x; "$@" ) 2>&1 | tee -a "${logfile}"

  echo "------------------------------------------------------------" | tee -a "${logfile}"
  echo "[${tag}] DONE $(date -Is)" | tee -a "${logfile}"

  make_summary "${logfile}" "${summaryfile}" "${tag}"
}

# Same as run_and_log, but allows env injection (used for AdaHiP sweep)
run_and_log_env() {
  local tag="$1"; shift
  local logfile="${LOG_DIR}/${tag}.log"
  local summaryfile="${SUMMARY_DIR}/${tag}.summary.txt"

  echo "============================================================" | tee "${logfile}"
  echo "[${tag}] $(date -Is)" | tee -a "${logfile}"
  echo "ENV: $*" | tee -a "${logfile}"
  echo "------------------------------------------------------------" | tee -a "${logfile}"

  ( set -x; env "$@" ) 2>&1 | tee -a "${logfile}"

  echo "------------------------------------------------------------" | tee -a "${logfile}"
  echo "[${tag}] DONE $(date -Is)" | tee -a "${logfile}"

  make_summary "${logfile}" "${summaryfile}" "${tag}"
}

# ------------------------------------------------------------
# Main execution loop
# ------------------------------------------------------------
for method in ${METHODS}; do
  BASE_ARGS=($(common_args "${method}"))

  if have_job "ppl"; then
    for dataset in "${DATASETS_PPL[@]}"; do
      for stride in "${STRIDES[@]}"; do

        if is_fa2 "${method}"; then
          tag="ppl_${dataset}_${method}_${MODEL}_s${stride}_$(ts)"
          run_and_log "${tag}" \
            uv run -m hip_research.main.model_eval \
              --job ppl \
              "${BASE_ARGS[@]}" \
              --dataset "${dataset}" \
              --stride "${stride}" \
              --count "${COUNT}" \

        elif is_adahip "${method}"; then
          for a in "${ALPHAS_ARR[@]}"; do
            for p in "${POWERS_ARR[@]}"; do
              for k in "${KS[@]}"; do
                tag="ppl_${dataset}_${method}_${MODEL}_s${stride}_k${k}_a$(fmt_num "$a")_p$(fmt_num "$p")_$(ts)"
                run_and_log_env "${tag}" \
                  "HIP_EXTEND=${HIP_EXTEND}" \
                  "ADAHIP_ALPHA=${a}" \
                  "ADAHIP_POWER=${p}" \
                  uv run -m hip_research.main.model_eval \
                    --job ppl \
                    "${BASE_ARGS[@]}" \
                    --dataset "${dataset}" \
                    --stride "${stride}" \
                    --count "${COUNT}" \
                    --k "${k}"
              done
            done
          done

        else
          for k in "${KS[@]}"; do
            tag="ppl_${dataset}_${method}_${MODEL}_s${stride}_k${k}_$(ts)"
            run_and_log "${tag}" \
              uv run -m hip_research.main.model_eval \
                --job ppl \
                "${BASE_ARGS[@]}" \
                --dataset "${dataset}" \
                --stride "${stride}" \
                --count "${COUNT}" \
                --k "${k}"
          done
        fi

      done
    done
  fi
done

echo
echo "All experiments finished."
echo "Logs      : ${LOG_DIR}"
echo "Summaries : ${SUMMARY_DIR}"
