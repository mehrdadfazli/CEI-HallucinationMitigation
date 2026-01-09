#!/usr/bin/env python3
import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import gc
import requests
from io import BytesIO
import math

from model_utils import load_model_and_processor, process_inputs
# from CEI_utils import generate_two_pass_dynamic  # your latest function (as provided)


# ------------------------------- CLI -----------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate CEI on MMHal-Bench")

    # Model/runtime
    p.add_argument("--model_type", type=str, default="instructblip",
                   choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--cache_dir", type=str, default=None,
               help="HF cache dir (optional). If None, uses HF defaults/HF_HOME/TRANSFORMERS_CACHE.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # MMHal IO
    p.add_argument("--input", type=str,
                    default=os.getenv("MMHAL_INPUT", "data/MMHal-Bench/response_template.json"),
                    help="Input JSON file (or set MMHAL_INPUT env var).")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON file (if None, auto-named under log_dir)")
    p.add_argument("--images_root", type=str,
                    default=os.getenv("MMHAL_IMAGES_ROOT", "data/MMHal-Bench/images"),
                    help="Images root for relative image_src (or set MMHAL_IMAGES_ROOT env var).")

    # Logging
    p.add_argument("--log_dir", type=str, default="./results/MMHal-Bench",
                   help="Directory for logs/results")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Process only first N items (debug)")

    # Decoding/budget
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--repetition_penalty", type=float, default=1.10)
    p.add_argument("--KV_cache", action="store_true", default=False,
                   help="Enable KV cache path for LLaVA/Next inside two-pass loop")

    # CEI knobs (match your generate_two_pass_dynamic signature)
    p.add_argument("--injection_layer", type=int, default=10)
    p.add_argument("--context_layer", type=int, default=-1)
    p.add_argument("--context_idx", type=int, default=-1)

    p.add_argument("--K_mass", type=int, default=40)
    p.add_argument("--start_layer", type=int, default=1)
    p.add_argument("--alpha_method", type=str, default="sigmoid",
                   choices=["sigmoid", "linear", "cosine"])
    p.add_argument("--alpha_max", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=0.20)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--tau_lo", type=float, default=0.10)
    p.add_argument("--tau_hi", type=float, default=0.30)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--topK_mass_start_layer", type=int, default=-1)

    p.add_argument("--delta", type=float, default=0.30,
                   help="Down-gate alpha on within-word (no word-start marker)")
    p.add_argument("--gamma", type=float, default=0.20,
                   help="Plausibility threshold relative to Pass-1 max prob")

    # Optional tracing (per-question JSONL of per-token decisions)
    p.add_argument("--trace_dir", type=str, default=None)

    return p

# ---------- α mapping ----------
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
    question_id=None,
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
            "question_id": question_id,
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

# ---------------------- Image loader (local path or URL) ----------------------
def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        resp = requests.get(image_file, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


# ---------------------- Get context embedding once per item -------------------
def get_context_embedding(raw_image, query, *, model, processor, model_type,
                          ctx_layer: int, ctx_idx: int):
    """
    Compute c = hidden_states[ctx_layer][0, ctx_idx, :]
    Uses a single forward pass with output_hidden_states=True.
    """
    with torch.no_grad():
        inputs = process_inputs(raw_image, query, processor, model_type)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        if model_type in ("llava", "llava-next"):
            hiddens = outputs["hidden_states"]
        else:
            # InstructBLIP-style nested outputs
            lmo = outputs["language_model_outputs"]
            hiddens = lmo["hidden_states"]

    layers = list(hiddens)
    L = len(layers)
    layer_idx = ctx_layer if ctx_layer >= 0 else (L + ctx_layer)
    if not (0 <= layer_idx < L):
        raise ValueError(f"context_layer={ctx_layer} out of range [0, {L-1}]")

    hs = layers[layer_idx]  # [1, T, D]
    T_tok = hs.shape[1]
    tok_idx = ctx_idx if ctx_idx >= 0 else (T_tok + ctx_idx)
    if not (0 <= tok_idx < T_tok):
        raise ValueError(f"context_idx={ctx_idx} out of range [0, {T_tok-1}]")

    return hs[0, tok_idx, :].detach()



def main():
    args = build_parser().parse_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_mmhal_cei_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(f"Using device: {device}")

    # Model names
    model_names = {
        "instructblip": "Salesforce/instructblip-vicuna-7b",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf",
    }

    # Load model & processor
    logger.info(f"Loading model: {args.model_type}")
    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    model.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("Model and processor loaded.")

    # Save run config
    exp_id = np.random.randint(1000, 9999)
    cfg_path = os.path.join(args.log_dir, f"{args.model_type}_{exp_id}_mmhal_config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved config → {cfg_path}")

    # Output path
    out_path = args.output or os.path.join(args.log_dir, f"{args.model_type}_{exp_id}.json")
    logger.info(f"Output → {out_path}")

    # Load MMHal JSON (list of dicts with keys like 'image_src', 'question', and 'model_answer' to be filled)
    with open(args.input, "r") as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]
        logger.info(f"Limited to first {args.limit} items.")

    # Process items
    for idx, item in enumerate(tqdm(data, desc="MMHal-CEI")):
        image_src = item.get("image_src", "")
        question  = item.get("question", "")

        # Resolve image path
        if image_src.startswith("http://") or image_src.startswith("https://"):
            image_path = image_src
        else:
            # if image_src is absolute, use it; else join with images_root
            image_path = image_src if os.path.isabs(image_src) else os.path.join(args.images_root, image_src)

        try:
            raw_image = load_image(image_path)
        except Exception as e:
            logger.error(f"[{idx}] Failed to open image {image_path}: {e}")
            item["model_answer"] = "Error"
            continue

        # Build prompt (stick to question; MMHal expects free-form answer)
        prompt = question

        # Compute per-item context embedding
        try:
            c = get_context_embedding(
                raw_image, prompt,
                model=model, processor=processor, model_type=args.model_type,
                ctx_layer=args.context_layer, ctx_idx=args.context_idx
            )
        except Exception as e:
            logger.error(f"[{idx}] Context embedding error: {e}")
            item["model_answer"] = "Error"
            continue

        # Optional per-item trace
        trace_path = None
        if args.trace_dir:
            os.makedirs(args.trace_dir, exist_ok=True)
            # Use a stable name if available
            basename = os.path.basename(image_src).replace("/", "_")
            trace_path = os.path.join(args.trace_dir, f"{idx}_{basename}.jsonl")

        # Run CEI generator
        try:
            cap64, cap512 = generate_two_pass_dynamic(
                raw_image, prompt,
                model=model,
                processor=processor,
                model_type=args.model_type,
                context_embedding=c,
                injection_layer=args.injection_layer,
                K_mass=args.K_mass,
                start_layer=args.start_layer,
                alpha_method=args.alpha_method,
                alpha_max=args.alpha_max,
                tau=args.tau,
                T=args.T,
                tau_lo=args.tau_lo,
                tau_hi=args.tau_hi,
                beta=args.beta,
                max_new_tokens=args.max_new_tokens,
                topK_mass_start_layer=args.topK_mass_start_layer,
                do_sample=args.do_sample,
                logger=logger,
                trace_path=trace_path,
                question_id=idx,
                delta=args.delta,
                gamma=args.gamma,
                repetition_penalty=args.repetition_penalty,
                KV_cache=args.KV_cache,
            )
            # MMHal answers are sentence-level; prefer the longer (bounded) decode
            answer = cap512.strip() if cap512.strip() else cap64.strip()
            item["model_answer"] = answer if answer else " "
        except Exception as e:
            logger.error(f"[{idx}] Generation error: {e}")
            item["model_answer"] = "Error"

        # Persist progress each step so you can monitor live
        try:
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush(); os.fsync(f.fileno())
        except Exception as e:
            logger.warning(f"[{idx}] Failed to write progress to {out_path}: {e}")

        torch.cuda.empty_cache(); gc.collect()

    logger.info(f"Done. Saved results to {out_path}")


if __name__ == "__main__":
    main()