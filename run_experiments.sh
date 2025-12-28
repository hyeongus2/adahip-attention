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
METHODS="${METHODS:-adahip}"

# JOBS:
# - ppl, passkey, stream
JOBS="${JOBS:-ppl}"

# ------------------------------------------------------------
# PPL experiment configuration
# ------------------------------------------------------------
# DATASETS_PPL:
# - wikitext / pg19
DATASETS_PPL=(${DATASETS_PPL:-pg19})

# STRIDES:
# - Context window stride for evaluation
# 32768, 65536, 131072
STRIDES=(${STRIDES:-65536})

# KS:
# - Attention budget (meaningful for hip/adahip)
# 256, 512, 1024
KS=(${KS:-512})

# COUNT:
# - Number of evaluation segments
# - -1 means "use full available segments"
COUNT="${COUNT:--1}"

COUNT_RAW="${COUNT}"
# [MOD] Dataset-specific base count policy when COUNT_RAW = -1.
# We will scale counts by stride so that evaluation "token budget" stays proportional:
#   COUNT = base_count(dataset) * (BASE_STRIDE / stride)
BASE_STRIDE="${BASE_STRIDE:-131072}"           # 128k reference stride

WIKITEXT_BASE_COUNT="${WIKITEXT_BASE_COUNT:-2}"  # observed: COUNT=-1 at 128k ~= 2 steps
PG19_BASE_COUNT="${PG19_BASE_COUNT:-10}"         # set smaller than full (e.g., 10~30) to fit runtime

MIN_COUNT="${MIN_COUNT:-1}"
MAX_COUNT="${MAX_COUNT:-100000}"               # optional hard cap (e.g., 200)

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
ALPHAS_STR="${ALPHAS:-1.0}"
POWERS_STR="${POWERS:-2.0}"

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

# [MOD] Compute effective COUNT for a given dataset and stride.
# - If COUNT_RAW != -1: use COUNT_RAW as a fixed count.
# - If COUNT_RAW == -1:
#     * wikitext: use WIKITEXT_BASE_COUNT at BASE_STRIDE and scale by stride
#     * pg19:     use PG19_BASE_COUNT at BASE_STRIDE and scale by stride
effective_count() {
  local dataset="$1"
  local stride="$2"

  # Honor explicit COUNT when COUNT_RAW is not -1
  if [[ "${COUNT_RAW}" != "-1" ]]; then
    echo "${COUNT_RAW}"
    return
  fi

  local base_count="-1"
  case "${dataset}" in
    wikitext) base_count="${WIKITEXT_BASE_COUNT}" ;;
    pg19)     base_count="${PG19_BASE_COUNT}" ;;
    *)        base_count="-1" ;;  # unknown dataset: fallback to full behavior
  esac

  # If base_count is -1, keep "full evaluation" semantics
  if [[ "${base_count}" == "-1" ]]; then
    echo "-1"
    return
  fi

  # Scale count inversely with stride (assumes stride divides BASE_STRIDE cleanly)
  local c=$(( base_count * BASE_STRIDE / stride ))

  if (( c < MIN_COUNT )); then c="${MIN_COUNT}"; fi
  if (( c > MAX_COUNT )); then c="${MAX_COUNT}"; fi
  echo "${c}"
}



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

        # [MOD] Update COUNT per dataset/stride without touching the run command lines.
        COUNT="$(effective_count "${dataset}" "${stride}")"

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

        # [MOD] Restore COUNT to the raw policy value after each stride iteration.
        COUNT="${COUNT_RAW}"

      done
    done
  fi
done

echo
echo "All experiments finished."
echo "Logs      : ${LOG_DIR}"
echo "Summaries : ${SUMMARY_DIR}"
