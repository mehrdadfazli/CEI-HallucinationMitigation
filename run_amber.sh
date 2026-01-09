#!/bin/bash

# Get the configuration file and log directory from arguments
CONFIG_FILE=$1
LOG_DIR=$2

# Helpers to read JSON
read_config() {
    python -c "import json; c=json.load(open('$CONFIG_FILE')); v=c.get('$1', ''); print(v if v is not None else '')"
}
read_array() {
    python -c "import json; c=json.load(open('$CONFIG_FILE')); a=c.get('$1', []); print(' '.join(map(str, a)))"
}

# Read values
model_type=$(read_config model_type)
load_in_8bit=$(read_config load_in_8bit)
cache_dir=$(read_config cache_dir)

use_CEI=$(read_config use_CEI)
do_sample=$(read_config do_sample)
num_beams=$(read_config num_beams)
max_new_tokens=$(read_config max_new_tokens)

context_embedding_idx=$(read_config context_embedding_idx)
context_embedding_layer=$(read_config context_embedding_layer)
injection_layer=$(read_config injection_layer)
alpha=$(read_config alpha)

dynamic_mode=$(read_config dynamic_mode)
alpha_method=$(read_config alpha_method)
K_mass=$(read_config K_mass)
start_layer=$(read_config start_layer)
topK_mass_start_layer=$(read_config topK_mass_start_layer)

tau=$(read_config tau)
T=$(read_config T)
tau_lo=$(read_config tau_lo)
tau_hi=$(read_config tau_hi)
beta=$(read_config beta)

amber_path=$(read_config amber_path)
queries_json=$(read_config queries_json)
annotations_json=$(read_config annotations_json)
image_dirname=$(read_config image_dirname)
random_seed=$(read_config random_seed)

KV_cache=$(read_config KV_cache)
delta=$(read_config delta)
repetition_penalty=$(read_config repetition_penalty)
gamma=$(read_config gamma)

# Basic validation
if [ -z "$model_type" ]; then
  echo "Error: Failed to read 'model_type' from $CONFIG_FILE"
  exit 1
fi

# Build the command
cmd="python src/run_AMBER.py \
  --model_type $model_type \
  --cache_dir \"$cache_dir\" \
  --log_dir \"$LOG_DIR\" \
  --num_beams $num_beams \
  --max_new_tokens $max_new_tokens \
  --context_embedding_idx $context_embedding_idx \
  --context_embedding_layer $context_embedding_layer \
  --injection_layer $injection_layer \
  --alpha $alpha \
  --dynamic_mode $dynamic_mode \
  --alpha_method $alpha_method \
  --K_mass $K_mass \
  --start_layer $start_layer \
  --tau $tau \
  --T $T \
  --tau_lo $tau_lo \
  --tau_hi $tau_hi \
  --beta $beta \
  --amber_path \"$amber_path\" \
  --queries_json \"$queries_json\" \
  --annotations_json \"$annotations_json\" \
  --image_dirname \"$image_dirname\" \
  --delta $delta \
  --gamma $gamma \
  --repetition_penalty $repetition_penalty \
  --random_seed $random_seed"

# Optional: only add if provided and not -1
if [ -n "$topK_mass_start_layer" ] && [ "$topK_mass_start_layer" != "-1" ]; then
  cmd="$cmd --topK_mass_start_layer $topK_mass_start_layer"
fi

# Add boolean flags conditionally
if [ "$load_in_8bit" = "true" ] || [ "$load_in_8bit" = "True" ]; then
  cmd="$cmd --load_in_8bit"
fi
if [ "$do_sample" = "true" ] || [ "$do_sample" = "True" ]; then
  cmd="$cmd --do_sample"
fi
if [ "$use_CEI" = "true" ] || [ "$use_CEI" = "True" ]; then
  cmd="$cmd --use_CEI"
fi
if [ "$opera_results" = "true" ] || [ "$opera_results" = "True" ]; then
  cmd="$cmd --opera_results"
fi
if [ "$KV_cache" = "true" ] || [ "$KV_cache" = "True" ]; then
  cmd="$cmd --KV_cache"
fi

# Show and run
echo "$cmd"
eval "$cmd"