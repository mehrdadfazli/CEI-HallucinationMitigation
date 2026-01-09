import argparse
import os
import json
import random
import torch
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import gc
import math

from model_utils import load_model_and_processor, process_inputs, model_names
from CEI_utils import setup_injection_hook

# Set up argument parser
parser = argparse.ArgumentParser(description="Run CEI on CHAIR benchmark")
parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava", "llava-next"], help="Model type")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default=None,
                    help="HF cache dir (optional). If None, uses HF_HOME/TRANSFORMERS_CACHE default.")
parser.add_argument("--log_dir", default="./results/CHAIR", help="Directory for logs and results")
parser.add_argument("--use_CEI", action="store_true", default=True, help="Use CEI for hallucination mitigation")
parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")

parser.add_argument("--context_embedding_idx", type=int, default=-1, help="Reference input token index for context embedding")
parser.add_argument("--context_embedding_layer", type=int, default=-1, help="Reference layer for context embedding")
parser.add_argument("--injection_layer", type=int, default=10, help="Layer to inject context embedding")
parser.add_argument("--alpha", type=float, default=0.1, help="Weighting factor for context embedding")

parser.add_argument("--opera_results_path", type=str, default=os.getenv("OPERA_RESULTS_PATH", None),
                    help="Path to OPERA JSONL (optional). Overrides internal defaults if provided.")
parser.add_argument("--chair_data_path", type=str, default=os.getenv("CHAIR_DATA_PATH", "data/coco2014/val2014"),
                    help="Path to COCO val2014 images (can also set CHAIR_DATA_PATH env var).")
parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

parser.add_argument("--dynamic_mode", default="two_pass", choices=["none", "two_pass"],
                    help="Dynamic CEI mode. 'two_pass' enables per-token two-forward strategy; 'none' = static CEI.")
parser.add_argument("--alpha_method", default="sigmoid", choices=["sigmoid", "linear", "cosine"],
                    help="Mapping from mean Top-K mass to alpha.")
parser.add_argument("--K_mass", type=int, default=40, help="K for Top-K mass (default 40).")
parser.add_argument("--start_layer", type=int, default=1, help="First layer index to include in tally (skip embeddings).")
parser.add_argument("--tau", type=float, default=0.2, help="Sigmoid center (used if alpha_method=sigmoid).")
parser.add_argument("--T", type=float, default=0.05, help="Sigmoid temperature (used if alpha_method=sigmoid).")
parser.add_argument("--tau_lo", type=float, default=0.10, help="Linear low breakpoint (used if alpha_method=linear).")
parser.add_argument("--tau_hi", type=float, default=0.30, help="Linear high breakpoint (used if alpha_method=linear).")
parser.add_argument("--topK_mass_start_layer", type=int, default=-1, help="If >=0, start layer for Top-K mass tally (overrides --start_layer for the tally only).")
parser.add_argument("--beta", type=float, default=0.30,
                    help="Cosine cutoff: alpha falls to 0 by m=beta (used when --alpha_method=cosine or cosine_raw).")

parser.add_argument("--trace_alpha", action="store_true", default=False,
                    help="If set, write per-token alpha traces to JSONL files.")
parser.add_argument("--trace_dir", type=str, default=None,
                    help="Directory to write trace JSONL files (defaults to <log_dir>/traces).")
parser.add_argument("--trace_samples", type=int, default=20,
                    help="Max number of images to trace (one JSONL per image).")

# Generation control
parser.add_argument("--repetition_penalty", type=float, default=1.1,
                    help=">1.0 penalizes previously generated tokens.")
parser.add_argument("--delta", type=float, default=0.5,
                    help="Scale α when next token is a suffix (not a word start).")
parser.add_argument("--gamma", type=float, default=0.0,
                    help="Keep Pass-2 top-1 only if it’s within gamma*max from Pass-1; else fallback to Pass-1 top-1.")
parser.add_argument("--KV_cache", action="store_true", default=False,
                    help="Use KV cache for LLaVA/LLaVA-Next.")


args = parser.parse_args()


def alpha_from_mass(
    m,
    method="sigmoid",
    *,
    alpha_max=0.1,
    tau=0.2,
    T=0.05,
    tau_lo=0.10,
    tau_hi=0.30,
    beta=0.30,
):
    """
    m: float in [0,1]  (we use layer-mean for mapping to keep tau/T interpretable)
    returns scalar alpha in [0, alpha_max]
    """
    import numpy as np
    m = float(m)
    if method == "sigmoid":
        p = 1.0 / (1.0 + np.exp(-(tau - m) / max(1e-12, T)))
        return alpha_max * p
    
    elif method == "linear":
        if tau_hi <= tau_lo:
            raise ValueError("Require tau_lo < tau_hi for linear mapping.")
        x = (tau_hi - m) / (tau_hi - tau_lo)
        x = max(0.0, min(1.0, x))
        return alpha_max * x
    
    elif method == "cosine":
        # Monotone half-cosine: alpha_max -> 0 by m = beta
        if beta <= 0:
            raise ValueError("beta must be > 0 for cosine method.")
        z = max(0.0, min(1.0, m / beta))          # normalize to [0,1]
        w = math.cos(0.5 * math.pi * z)           # 1 -> 0 as z goes 0->1
        return alpha_max * w

    else:
        raise ValueError("method must be 'sigmoid' or 'linear' or 'cosine'")


# ---------- Two-pass dynamic CEI (identical behavior to run_CHAIR) ----------
def generate_two_pass_dynamic(
    raw_image,
    query,
    *,
    model,
    processor,
    model_type,
    context_embedding,
    injection_layer,
    K_mass=40,
    start_layer=1,
    alpha_method="sigmoid",
    alpha_max=0.1,
    tau=0.2,
    T=0.05,
    tau_lo=0.10,
    tau_hi=0.30,
    beta=0.30,                 
    max_new_tokens=512,
    topK_mass_start_layer=-1,
    do_sample=False,           
    logger=None,
    trace_path=None,
    image_id=None,
    delta=0.3,
    gamma=0.2,
    repetition_penalty=1.1,
    KV_cache=False,

):
    """
    Two-pass per token:
      Pass 1: probe (no injection) -> final Top-K -> tally Top-K mass across selected layers -> mean -> alpha
              * step 0: full prefix, cache OFF
              * step >=1: single token + KV cache (safe on HF 4.47)
      Pass 2: inject with alpha -> greedy next token; advance KV cache for LLaVA/LLaVA-Next
      NEW:
        - Down-gate alpha on within-word tokens (detected from Pass-1 top-1 token).
        - Plausibility constraint: if Pass-2 top-1 ∉ {i | p1[i] ≥ γ·max(p1)}, fallback to Pass-1 top-1.
    """
    import torch
    import torch.nn.functional as F
    import math
    from CEI_utils import setup_injection_hook

    model.eval()

    # optional tracing
    tf = None
    trace_meta = None
    if trace_path is not None:
        tf = open(trace_path, "a")
        trace_meta = {
            "image_id": image_id,
            "model_type": model_type,
            "injection_layer": injection_layer,
            "K_mass": K_mass,
            "start_layer": start_layer,
        }

    supports_kv_cache = (model_type in ("llava", "llava-next"))
    supports_kv_cache = supports_kv_cache and KV_cache

    # lm_head for logit lens
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "lm_head"):
        raise RuntimeError("Expected `model.language_model.lm_head` for logit-lens.")
    lm_head = lm.lm_head

    # Build base inputs once
    inputs_base = process_inputs(raw_image, query, processor, model_type)
    generated = inputs_base["input_ids"]
    input_length = generated.shape[-1]

    attention_mask = inputs_base.get("attention_mask", None)
    if attention_mask is None:
        bsz, seq_len = generated.shape
        attention_mask = torch.ones((bsz, seq_len), device=generated.device, dtype=torch.long)

    current_input_ids = generated
    past_key_values = None
    eos_id = processor.tokenizer.eos_token_id

    # helpers
    def build_step_inputs(current_ids, attn_mask):
        step_inputs = {k: v for k, v in inputs_base.items()}
        step_inputs["input_ids"] = current_ids
        step_inputs["attention_mask"] = attn_mask
        return step_inputs

    def _convert_id_to_token_str(tid: int) -> str:
        tok = None
        if hasattr(processor.tokenizer, "convert_ids_to_tokens"):
            tok = processor.tokenizer.convert_ids_to_tokens([tid])[0]
        else:
            tok = processor.tokenizer.decode([tid], skip_special_tokens=False)
        return tok or ""

    def _is_word_start(tok_str: str) -> bool:
        # SentencePiece: '▁' denotes word start; BPE (GPT): 'Ġ' denotes space-before
        return tok_str.startswith("▁") or tok_str.startswith("Ġ") or tok_str.startswith(" ")

    with torch.no_grad():
        for step in range(max_new_tokens):
            # ---------- PASS 1: probe ----------
            if supports_kv_cache and past_key_values is not None:
                # step >= 1: probe single token with cache
                probe_input_ids = current_input_ids
                inputs_p1 = build_step_inputs(probe_input_ids, attention_mask)
                out1 = model(
                    **inputs_p1,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            else:
                # step 0 or non-cached models: full prefix, cache OFF
                probe_input_ids = generated
                inputs_p1 = build_step_inputs(probe_input_ids, attention_mask)
                if supports_kv_cache:
                    out1 = model(**inputs_p1, output_hidden_states=True, return_dict=True, use_cache=False)
                else:
                    out1 = model(**inputs_p1, output_hidden_states=True, return_dict=True)

            # unify hidden states + final logits
            if model_type in ("llava", "llava-next"):
                hidden_states = out1["hidden_states"]
                final_logits = out1.logits[:, -1, :]
            else:
                lm_out = out1["language_model_outputs"]
                hidden_states = lm_out["hidden_states"]
                final_logits = lm_out.logits[:, -1, :]

            final_probs = F.softmax(final_logits, dim=-1)              # [B, V]
            top1_id_p1 = torch.argmax(final_probs, dim=-1)             # [B]
            topk_idx = torch.topk(final_probs, k=K_mass, dim=-1).indices

            # Tally Top-K mass across selected layers (use final Top-K ids)
            masses = []
            L = len(hidden_states)
            layer_start = topK_mass_start_layer if (topK_mass_start_layer is not None and topK_mass_start_layer >= 0) else start_layer
            for layer_idx in range(layer_start, L):                    # FIX: use layer_start
                hs = hidden_states[layer_idx][:, -1, :]
                logits_l = lm_head(hs).float()
                probs_l = F.softmax(logits_l, dim=-1)
                mass_l = probs_l.gather(-1, topk_idx).sum(dim=-1)      # [B]
                masses.append(mass_l)

            masses = torch.stack(masses, dim=0) if len(masses) > 0 else torch.zeros((1, final_probs.shape[0]), device=final_probs.device)
            mean_mass = masses.mean(dim=0)                              # [B]
            sum_mass  = masses.sum(dim=0)                               # [B]

            m = float(mean_mass[0].item())
            a = alpha_from_mass(
                m,
                method=alpha_method,
                alpha_max=alpha_max,
                tau=tau,
                T=T,
                tau_lo=tau_lo,
                tau_hi=tau_hi,
                beta=beta,
            )

            # ---- NEW: down-gate alpha on within-word tokens (based on Pass-1 top-1 token) ----
            tok1_text = _convert_id_to_token_str(int(top1_id_p1[0].item()))
            if not _is_word_start(tok1_text):
                a = a * float(delta)

            # ---- NEW: plausibility set from Pass-1 ----
            max_p1 = final_probs.max(dim=-1, keepdim=True).values      # [B,1]
            thresh = float(gamma) * max_p1                        # [B,1]
            plausible_mask = final_probs >= thresh                      # [B,V] boolean
            # store plausible ids for batch item 0
            plausible_ids = torch.nonzero(plausible_mask[0], as_tuple=False)[:, 0]  # [M]
            plausible_set = set(plausible_ids.tolist())

            # ---------- PASS 2: inject + pick next token ----------
            hook_handle = setup_injection_hook(model, injection_layer, context_embedding, a)
            try:
                inputs_p2 = build_step_inputs(current_input_ids, attention_mask)
                if supports_kv_cache:
                    out2 = model(**inputs_p2, return_dict=True, use_cache=True, past_key_values=past_key_values)
                else:
                    out2 = model(**inputs_p2, return_dict=True)
            finally:
                hook_handle.remove()


            # Repetition penalty (applied to Pass-2 logits BEFORE choosing next token)
            last_logits = out2.logits[:, -1, :]                   # [1, V]
            if repetition_penalty is not None and repetition_penalty > 1.0:
                hist = generated[:, input_length:]                 # previously generated (exclude prompt)
                if hist.numel() > 0:
                    uniq = torch.unique(hist[0])
                    # In-place friendly adjustment per Keskar et al. (CTRL)
                    #   pos logits -> divide by penalty; neg logits -> multiply by penalty
                    lv = last_logits[0]
                    for tid in uniq.tolist():
                        val = lv[tid]
                        lv[tid] = val / repetition_penalty if val > 0 else val * repetition_penalty


            # last_logits = out2.logits[:, -1, :]
            next_id_p2 = int(torch.argmax(last_logits, dim=-1)[0].item())

            # ---- NEW: plausibility constraint decision ----
            if next_id_p2 in plausible_set:
                chosen_id = next_id_p2
                chosen_source = "pass2"
            else:
                chosen_id = int(top1_id_p1[0].item())
                chosen_source = "pass1_fallback"

            # prevent immediate duplicate token (minimal, local)
            prev_id = int(generated[0, -1].item())
            if chosen_id == prev_id:
                # try second-best from Pass-2 that's also plausible; else fall back to Pass-1 top-1
                vals, idxs = torch.topk(last_logits[0], k=5)  # small beam over logits
                picked = None
                for cand in idxs.tolist():
                    if cand != prev_id and (cand in plausible_set):
                        picked = cand; break
                if picked is None:
                    picked = int(top1_id_p1[0].item())
                chosen_id = picked


            next_token = torch.tensor([[chosen_id]], device=generated.device, dtype=generated.dtype)

            # trace (optional)
            if tf is not None:
                tok_text = processor.tokenizer.decode([chosen_id], skip_special_tokens=False)
                record = {
                    **trace_meta,
                    "step_idx": step,
                    "is_eos": bool(chosen_id == eos_id),
                    "token_id": chosen_id,
                    "token_text": tok_text,
                    "alpha": float(a),
                    "mean_topK_mass": float(m),
                    "sum_topK_mass": float(sum_mass[0].item()),
                    "within_word_gate_applied": bool(not _is_word_start(tok1_text)),
                    "gamma": float(gamma),
                    "choice": chosen_source,
                    "repetition_penalty": float(repetition_penalty),
                }
                json.dump(record, tf); tf.write("\n"); tf.flush()

            # advance cache from Pass-2 forward (authoritative state)
            if supports_kv_cache:
                past_key_values = out2.past_key_values

            # append token
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=-1
            )
            current_input_ids = next_token if supports_kv_cache else generated

            # EOS check
            if chosen_id == eos_id:
                break

            torch.cuda.empty_cache()

    gen_only = generated[:, input_length:]
    gen_64 = gen_only[:, :64]
    gen_512 = gen_only[:, :512]
    caption_64 = processor.batch_decode(gen_64, skip_special_tokens=True)[0].strip()
    caption_512 = processor.batch_decode(gen_512, skip_special_tokens=True)[0].strip()

    if tf is not None:
        tf.close()

    return caption_64, caption_512




def load_opera_image_ids(opera_results_path):
    image_ids = []
    try:
        with open(opera_results_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_ids.append(data['image_id'])
        return image_ids
    except Exception as e:
        logging.error(f"Error reading OPERA results file {opera_results_path}: {e}")
        raise


def main():
    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"use_CEI is set to: {args.use_CEI}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Experiment configuration
    exp_config = {
        "context_embedding_idx": args.context_embedding_idx,
        "context_embedding_layer": args.context_embedding_layer,
        "injection_layer": args.injection_layer,
        "alpha": args.alpha,
        "max_new_tokens": args.max_new_tokens,
        "dynamic_mode": args.dynamic_mode,
        "alpha_method": args.alpha_method,
        "K_mass": args.K_mass,
        "start_layer": args.start_layer,
        "topK_mass_start_layer": args.topK_mass_start_layer,
        "tau": args.tau,
        "T": args.T,
        "tau_lo": args.tau_lo,
        "tau_hi": args.tau_hi,
        "beta": args.beta,
        "repetition_penalty": args.repetition_penalty,
    }

    # Generate random exp_id
    exp_id = np.random.randint(1000, 9999)

    # Save experiment configuration
    config_path = os.path.join(args.log_dir, f"config_{exp_id}.json")
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=4)
    logger.info(f"Saved experiment configuration to {config_path}")

    # Set up output file
    output_file = os.path.join(args.log_dir, f"{args.model_type}_{exp_id}.jsonl")
    logger.info(f"Output will be saved to {output_file}")

    # --- Trace setup (optional) ---
    if args.trace_dir is None:
        trace_root = os.path.join(args.log_dir, "traces")
    else:
        trace_root = args.trace_dir
    if args.trace_alpha:
        os.makedirs(trace_root, exist_ok=True)
    n_traced = 0  # how many images we've traced so far

    # Load model and processor
    model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)

    
    if args.opera_results:
        if args.model_type == "llava":
            opera_results_path = "path/to/OPERA/llava/ours.jsonl"
        elif args.model_type == "instructblip":
            opera_results_path = "path/to/OPERA/instructblip/ours.jsonl"

        image_ids = load_opera_image_ids(opera_results_path)
        image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]
        logger.info(f"Loaded {len(image_list)} image IDs from OPERA results")
    else:
        # raise NotImplementedError("Failed to load images from OPERA results")
        img_files = [f for f in os.listdir(args.chair_data_path) if f.endswith('.jpg')]
        image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
        random.seed(args.random_seed)
        random.shuffle(image_list)
        if args.num_images is not None:
            image_list = image_list[:args.num_images]
        logger.info(f"Processing {len(image_list)} images from {args.chair_data_path}")    
    

    # Process each image
    for img_file, img_id in tqdm(image_list, desc="Processing images"):
        img_path = os.path.join(args.chair_data_path, img_file)
            
        raw_image = Image.open(img_path).convert('RGB')
        query = "Describe this image."

        context_embedding = None
        if args.use_CEI:
            # Get context embedding
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["attention_mask"] = torch.ones((batch_size, seq_len), device=inputs["input_ids"].device, dtype=torch.long)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            if args.model_type in ("llava", "llava-next"):
                hidden_states = outputs['hidden_states']
            else:
                hidden_states = outputs['language_model_outputs']['hidden_states']
            
            context_embedding = hidden_states[exp_config["context_embedding_layer"]][0, exp_config["context_embedding_idx"], :]

        # ---- Generation ----
        # Decide if we trace this image
        do_trace = args.trace_alpha and (n_traced < args.trace_samples)
        trace_path = None
        if do_trace:
            trace_path = os.path.join(trace_root, f"{exp_id}_{args.model_type}_{img_id}.trace.jsonl")

        # ---- Generation ----
        if args.use_CEI and args.dynamic_mode == "two_pass":
            # Dynamic (two-forward-pass per token)
            caption_64, caption_512 = generate_two_pass_dynamic(
                raw_image, query,
                model=model,
                processor=processor,
                model_type=args.model_type,
                context_embedding=context_embedding,
                injection_layer=exp_config["injection_layer"],
                K_mass=args.K_mass,
                start_layer=args.start_layer,
                topK_mass_start_layer=args.topK_mass_start_layer,
                alpha_method=args.alpha_method,
                alpha_max=exp_config["alpha"],
                tau=args.tau, T=args.T,
                tau_lo=args.tau_lo, tau_hi=args.tau_hi,
                max_new_tokens=exp_config["max_new_tokens"],
                do_sample=args.do_sample,
                logger=logging.getLogger(__name__),
                beta=args.beta,
                repetition_penalty=args.repetition_penalty,
                delta=args.delta,
                gamma=args.gamma,
                KV_cache=args.KV_cache,
            )
            if do_trace:
                n_traced += 1
        else:
            # ---- Static CEI (your original path) ----
            if args.use_CEI:
                hook_handle = setup_injection_hook(model, exp_config["injection_layer"], context_embedding, exp_config["alpha"])

            # Process inputs for generation
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            # Generate caption
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    do_sample=args.do_sample, 
                    max_new_tokens=exp_config["max_new_tokens"], 
                    num_beams=args.num_beams, 
                    repetition_penalty=args.repetition_penalty
                )
            generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
            
            caption_64 = processor.batch_decode(generated_ids[:, :64], skip_special_tokens=True)[0].strip()
            caption_512 = processor.batch_decode(generated_ids[:, :512], skip_special_tokens=True)[0].strip()

            if args.use_CEI:
                hook_handle.remove()

        # ---- Save result (unchanged) ----
        result = {"image_id": img_id, "caption_64": caption_64, "caption_512": caption_512}
        with open(output_file, "a") as f:
            json.dump(result, f); f.write('\n')

        torch.cuda.empty_cache(); gc.collect()

    logger.info(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()