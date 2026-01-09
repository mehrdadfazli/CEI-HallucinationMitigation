#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_mmhal_sink_cei.sh configs/mmhal_llava.json [LOG_DIR_OVERRIDE]

CONFIG_FILE="$1"
LOG_DIR_OVERRIDE="${2:-}"

# ---------- Helpers ----------
read_config() {
  python - "$CONFIG_FILE" "$1" << 'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
v = cfg.get(sys.argv[2], "")
print("" if v is None else v)
PY
}

# ---------- Read fields (names match run_MMHal_CEI.py) ----------
# Model / runtime
model_type=$(read_config model_type)
gpu_id=$(read_config gpu_id)
load_in_8bit=$(read_config load_in_8bit)
cache_dir=$(read_config cache_dir)
device=$(read_config device)

# IO
# input_json=$(read_config input)
output_json=$(read_config output)
images_root=$(read_config images_root)
log_dir=$(read_config log_dir)
random_seed=$(read_config random_seed)
limit=$(read_config limit)

# Decode / budget
do_sample=$(read_config do_sample)
max_new_tokens=$(read_config max_new_tokens)
repetition_penalty=$(read_config repetition_penalty)
KV_cache=$(read_config KV_cache)

# CEI knobs
injection_layer=$(read_config injection_layer)
context_layer=$(read_config context_layer)
context_idx=$(read_config context_idx)

K_mass=$(read_config K_mass)
start_layer=$(read_config start_layer)
alpha_method=$(read_config alpha_method)
alpha_max=$(read_config alpha_max)
tau=$(read_config tau)
Tval=$(read_config T)          # avoid clobbering 'T' in shell
tau_lo=$(read_config tau_lo)
tau_hi=$(read_config tau_hi)
beta=$(read_config beta)
topK_mass_start_layer=$(read_config topK_mass_start_layer)

delta=$(read_config delta)
gamma=$(read_config gamma)
trace_dir=$(read_config trace_dir)

# ---------- Validate ----------
if [ -z "$model_type" ]; then
  echo "Error: 'model_type' missing in $CONFIG_FILE" >&2; exit 1
fi
# if [ -z "$input_json" ]; then
#   echo "Error: 'input' (MMHal JSON) missing in $CONFIG_FILE" >&2; exit 1
# fi

# Optional override
if [ -n "$LOG_DIR_OVERRIDE" ]; then
  log_dir="$LOG_DIR_OVERRIDE"
fi
[ -z "$log_dir" ] && log_dir="./results/MMHal-Bench"

# ---------- Build command ----------
cmd="python src/run_MMHal.py \
  --model_type $model_type \
  --gpu_id ${gpu_id:-0} \
  --images_root \"${images_root:-./datasets/MMHal-Bench/images}\" \
  --log_dir \"$log_dir\" \
  --max_new_tokens ${max_new_tokens:-128} \
  --repetition_penalty ${repetition_penalty:-1.1} \
  --injection_layer ${injection_layer:-10} \
  --context_layer ${context_layer:--1} \
  --context_idx ${context_idx:--1} \
  --K_mass ${K_mass:-40} \
  --start_layer ${start_layer:-1} \
  --alpha_method ${alpha_method:-sigmoid} \
  --alpha_max ${alpha_max:-0.10} \
  --tau ${tau:-0.20} \
  --T ${Tval:-0.05} \
  --tau_lo ${tau_lo:-0.10} \
  --tau_hi ${tau_hi:-0.30} \
  --beta ${beta:-0.30} \
  --delta ${delta:-0.30} \
  --gamma ${gamma:-0.20}"

# Optional args if provided
[ -n "$output_json" ] && cmd="$cmd --output \"$output_json\""
[ -n "$cache_dir" ]   && cmd="$cmd --cache_dir \"$cache_dir\""
[ -n "$device" ]      && cmd="$cmd --device \"$device\""
[ -n "$trace_dir" ]   && cmd="$cmd --trace_dir \"$trace_dir\""
[ -n "$random_seed" ] && cmd="$cmd --random_seed $random_seed"
[ -n "$limit" ]       && cmd="$cmd --limit $limit"

# topK_mass_start_layer only if set and not -1
if [ -n "$topK_mass_start_layer" ] && [ "$topK_mass_start_layer" != "-1" ]; then
  cmd="$cmd --topK_mass_start_layer $topK_mass_start_layer"
fi

# Boolean flags
case "$load_in_8bit" in true|True) cmd="$cmd --load_in_8bit" ;; esac
case "$do_sample"    in true|True) cmd="$cmd --do_sample" ;; esac
case "$KV_cache"     in true|True) cmd="$cmd --KV_cache" ;; esac

# ---------- Run ----------
echo "$cmd"
eval "$cmd"
