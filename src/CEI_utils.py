def setup_injection_hook(model, injection_layer, context_embedding, alpha, normalize_context=False, eps=1e-6):
    def injection_hook(module, input, output):
        hidden_states = output[0].clone()           # [B,T,H]
        hs_last = hidden_states[:, -1, :]           # [B,H]
        ctx = context_embedding.to(hs_last.device).type(hs_last.dtype)
        ctx = ctx.unsqueeze(0).expand_as(hs_last)   # [B,H]

        if normalize_context:
            hs_norm  = hs_last.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            ctx_norm = ctx.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            ctx = ctx * (hs_norm / ctx_norm)

        hidden_states[:, -1, :] = (1.0 - alpha) * hs_last + alpha * ctx
        return (hidden_states, output[1])

    return model.language_model.model.layers[injection_layer].register_forward_hook(injection_hook)

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
