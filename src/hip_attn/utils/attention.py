import os
import warnings
import math
import nvtx
import torch

from hip_attn.utils.attn_l1_loss import compute_attn_lp_loss_triton
from hip_attn.v1_0.attention1_block_gpu import flash_attention, hip_attention

# =========================================================================
# Helper Functions for AdaHiP (Dynamic-k) & Alignment
# =========================================================================

def _round_up_to_64(x: int) -> int:
    """Ensure k is a multiple of 64 for HiP kernels."""
    return ((x + 63) // 64) * 64

@torch.no_grad()
def _estimate_entropy_norm(q, k, probe_q=128, probe_k=1024):
    """
    Estimate attention entropy to determine dynamic k.
    """
    N, Tq, Hq, D = q.shape
    _, Tk, Hk, _ = k.shape

    Q = min(int(probe_q), int(Tq))
    P = min(int(probe_k), int(Tk))

    if Q <= 0 or P <= 1:
        return 0.0

    # Probe latest tokens
    curr_q = q[:, -Q:].to(torch.float32)
    curr_k = k[:, -P:].to(torch.float32)

    # GQA Handling
    if Hq != Hk:
        curr_k = curr_k.repeat_interleave(Hq // Hk, dim=2)

    # Compute local attention entropy
    logits = torch.einsum("nqhd,nphd->nqhp", curr_q, curr_k)
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=-1)

    # Normalize by max possible entropy (log P)
    ent_norm = ent / (math.log(P) + 1e-8)
    return float(ent_norm.mean().item())

# =========================================================================
# Main Attention Function
# =========================================================================

@nvtx.annotate("custom_attention")
def custom_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    causal_mask,
    attention_dropout,
    # Attention method
    attention_method="hip",
    tree_reformer=None,
    tree_performer=None,
    # hip parameters
    tree_k=512,
    tree_block_size_q=64,
    tree_block_stride_q=2,
    tree_block_size_k=2,
    tree_block_stride_k=1,
    tree_dense_queries=0,
    tree_last_dense_queries=0,
    tree_sampling_method="center",
    # Latency optimization tweaks
    tree_enable_flash=True,
    tree_enable_sparq=False,
    tree_use_sliding_window=True,
    tree_sliding_window_size=int(os.getenv("HIP_DRAFT_SLIDING_WINDOW", "1024")),
    tree_sink_token_size=256,
    # Context averaging parameters
    tree_using_context_avg=False,
    tree_avgpool_scaler=None,
    last_cumsum=None,
    hidden_states=None,
    # RoPE parameters
    tree_rope_method="none",
    need_apply_rope=False,
    rope_cos=None,
    rope_sin=None,
    position_ids=None,
    self_extend_group_size=4,
    # Attention sparsity loss
    output_attn_sparsity_loss=False,
    tree_lp_norm_coeff=0.5,
    # Hyper attention state
    hyper_attention=None,
    sm_scaler=None,
    attn_logit_softcapping=0,
    model_sliding_window=None,
    model_context_length=131072,
    layer_idx=10,
    extend_stages=None,
    sliding_window_indices=None,
    using_extend=False,
):
    if sm_scaler is None:
        sm_scaler = 1 / (query_states.shape[-1] ** 0.5)
    attn_sparsity_loss = None
    if model_sliding_window is not None:
        assert isinstance(model_sliding_window, int)

    N, H, T, HID = query_states.shape
    _N, _H, _T, _HID = key_states.shape
    assert (H % _H) == 0
    H_KV = _H
    last_cumsum = attn_sparsity_loss = None
    is_prompt = (N, T, HID) == (_N, _T, _HID)

    # --------------------------------------------------------------------------
    # 1. Standard Attention Methods (FA2, SDPA, etc.)
    # --------------------------------------------------------------------------
    if attention_method in ["none", "sdpa", "fa2"]:
        if query_states.device.type == "cuda" and attention_method == "sdpa":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        from flash_attn import flash_attn_func, flash_attn_with_kvcache

        if is_prompt:
            if attention_method in ["none", "fa2"]:
                if causal_mask is not None:
                    warnings.warn(f"causal mask provided. this is useless {causal_mask.shape}")
                attn_output = flash_attn_func(
                    q=query_states.permute(0, 2, 1, 3),
                    k=key_states.permute(0, 2, 1, 3),
                    v=value_states.permute(0, 2, 1, 3),
                    softmax_scale=sm_scaler,
                    causal=True,
                    softcap=attn_logit_softcapping,
                    window_size=((model_sliding_window, model_sliding_window) if model_sliding_window is not None else (-1, -1)),
                ).permute(0, 2, 1, 3)
            elif attention_method in ["spda", "sdpa"]:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=causal_mask, is_causal=causal_mask is None, dropout_p=attention_dropout,
                )
        else:
            if attention_method in ["none", "fa2"]:
                attn_output = flash_attn_with_kvcache(
                    q=query_states.permute(0, 2, 1, 3),
                    k_cache=key_states.permute(0, 2, 1, 3),
                    v_cache=value_states.permute(0, 2, 1, 3),
                    softmax_scale=sm_scaler,
                    causal=True,
                    softcap=attn_logit_softcapping,
                    window_size=((model_sliding_window, model_sliding_window) if model_sliding_window is not None else (-1, -1)),
                ).permute(0, 2, 1, 3)
            elif attention_method in ["sdpa"]:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=causal_mask, is_causal=causal_mask is None, dropout_p=attention_dropout,
                )

    # --------------------------------------------------------------------------
    # 2. HiP / Tree / AdaHiP Attention (The core logic)
    # --------------------------------------------------------------------------
    elif attention_method in ["hip", "tree", "adahip"]:
        q = query_states * sm_scaler
        k = key_states
        v = value_states

        N, H, TDST, HID = q.shape
        _, _, TSRC, _ = k.shape

        LAST_DENSE_QUERIES = tree_last_dense_queries
        if LAST_DENSE_QUERIES == 0: LAST_DENSE_QUERIES = None
        current_query_index = TSRC - TDST
        attn_outputs = []

        try:
            # 2-A. Legacy Path (v1.0)
            if os.getenv("HIP_LEGACY", "0") == "1":
                q = q.reshape(N * H, TDST, HID)
                k = k.reshape(N * H_KV, TSRC, HID)
                v = v.reshape(N * H_KV, TSRC, HID)
                q_hip = q[:, :, :]

                attn_output_hip, _ = hip_attention(
                    q_hip, k[:, :LAST_DENSE_QUERIES, :], v[:, :LAST_DENSE_QUERIES, :],
                    mask_k=tree_k, block_size_q=tree_block_size_q, block_size_k=tree_block_size_k,
                    dense_queries_exp=0, rope_method=tree_rope_method,
                    rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                    rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                    position_ids=position_ids, enable_sparq=False, is_flash=True,
                    using_sliding_window=True, sampling_method=tree_sampling_method, num_sink=16,
                )

            # 2-B. New Path (v1.2 Extend) - Dynamic K Logic Applied Here
            else:
                from hip_attn.v1_2.attention_extend import HiPAttentionArgs as HiPAttentionArgs12
                from hip_attn.v1_2.attention_extend import ScanStage
                from hip_attn.v1_2.attention_extend import dual_stage_quadratic_hip_attention as dual_stage_quadratic_hip_attention_extend

                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                # --- Dynamic K Calculation Logic ---
                base_k = int(tree_k)

                if attention_method == "adahip":
                    # AdaHiP: adjust k based on entropy
                    # Defaults keep behavior close to current ACR block, but keyed off ADAHIP_* env vars.
                    k_min = max(64, base_k // 4)
                    k_max = base_k * 2

                    # Prefer ADAHIP_*; fall back to legacy ACR_* if present
                    probe_q = int(os.getenv("ADAHIP_PROBE_Q", os.getenv("ACR_PROBE_Q", "128")))
                    probe_k = int(os.getenv("ADAHIP_PROBE_K", os.getenv("ACR_PROBE_K", "1024")))
                    alpha = float(os.getenv("ADAHIP_ALPHA", os.getenv("ACR_ALPHA", "1.0")))
                    power = float(os.getenv("ADAHIP_POWER", os.getenv("ACR_POWER", "2.0")))

                    ent = _estimate_entropy_norm(q, k, probe_q=probe_q, probe_k=probe_k)

                    # Normalize entropy to scaling factor x in [0, 1]
                    x = max(0.0, min(1.0, alpha * ent))
                    x = x ** power

                    target_k = int(k_min + (k_max - k_min) * x)
                else:
                    # Standard HiP: trust the user input
                    target_k = base_k

                # CRITICAL: Ensure alignment to 64
                target_k = _round_up_to_64(max(64, target_k))

                # --- Scale Stages (Ratios: 16 -> 4 -> 1) ---
                stage_1_k = target_k * 16
                stage_2_k = target_k * 4
                final_k = target_k

                # Setup Stages
                if extend_stages is None:
                    chosen_stages = [
                        ScanStage(stage_block_size_q=64, stage_block_stride_q=4, stage_chunk_size=128, stage_k=None, stage_stride=1),
                        ScanStage(stage_block_size_q=64, stage_block_stride_q=4, stage_chunk_size=32, stage_k=stage_1_k, stage_stride=1),
                        ScanStage(stage_block_size_q=64, stage_block_stride_q=1, stage_chunk_size=8, stage_k=stage_2_k, stage_stride=1),
                    ]
                    chosen_second_stage_k = final_k
                else:
                    # If external config is provided, respect it
                    if isinstance(extend_stages["stages"][0], dict):
                        for i in range(len(extend_stages["stages"])):
                            extend_stages["stages"][i] = ScanStage(**extend_stages["stages"][i])
                    chosen_stages = extend_stages["stages"]
                    chosen_second_stage_k = extend_stages["second_stage_k"]

                # Setup Arguments
                mask_only = False
                k_group_size = int(os.getenv("K_GROUP_SIZE", "1"))

                dual_stage_kwargs = dict(
                    q=q, k=k, v=v,
                    args=HiPAttentionArgs12(
                        mask_k=final_k,
                        block_size_k=64,
                        sliding_window_size=((1024 if layer_idx > 2 else 4096) if extend_stages is None else extend_stages.get("sliding_window_size", 4096)),
                        sink_token_size=(256 if extend_stages is None else extend_stages.get("sink_token_size", 256)),
                        using_extend=using_extend,
                        rope_cos=rope_cos.squeeze(0) if rope_cos is not None else None,
                        rope_sin=rope_sin.squeeze(0) if rope_sin is not None else None,
                        need_apply_rope=using_extend,

                        second_stage_k=chosen_second_stage_k,
                        stages=chosen_stages,

                        block_sparse_block_size_q=64,
                        model_context_length=model_context_length,
                        scan_extend_backend=("streaming" if layer_idx < 3 else "relative"),
                        sa_extend_backend=("streaming" if extend_stages is None else extend_stages.get("sa_extend_backend", "streaming")),
                        stage_early_terminate=k_group_size,
                        mask_only=mask_only,
                        require_stage_caches=False,
                        require_cache_statistics=False,
                        sliding_window_indices=sliding_window_indices,
                        layer_id=layer_idx,
                    ),
                )

                attn_output_hip, metadata = dual_stage_quadratic_hip_attention_extend(**dual_stage_kwargs)
                attn_output_hip = attn_output_hip.permute(0, 2, 1, 3)

        except RuntimeError as ex:
            os.makedirs("cache/hip", exist_ok=True)
            torch.save({"q": q, "k": k, "v": v}, "cache/hip/qkv.pth")
            raise Exception("oops hip is dead, check cache/hip/qkv.pth") from ex

        # Context Averaging (Legacy Feature)
        if tree_using_context_avg:
            if last_cumsum is None:
                last_cumsum = v.cumsum(-2, dtype=torch.float32)
                last_cumsum = last_cumsum[:, TSRC - TDST : LAST_DENSE_QUERIES, :]
            else:
                last_cumsum = last_cumsum.flatten(0, 1)
                curr_v = v[:, -q.shape[-2] : LAST_DENSE_QUERIES, :]
                curr_v = curr_v.cumsum(-2, dtype=torch.float32)
                last_cumsum = curr_v + last_cumsum[:, -1:, :]

            context_avg = (last_cumsum / torch.arange(
                current_query_index + 1,
                current_query_index + 1 + q.shape[1],
                device=v.device
            )[None, :, None]).to(v.dtype)

            last_cumsum = last_cumsum.unflatten(0, (N, H))
            scale_avg = (
                torch.sigmoid(tree_avgpool_scaler(hidden_states[:, :LAST_DENSE_QUERIES, :]).transpose(-1, -2).reshape(N * H, -1, 1))
                * 0.25
                * torch.clamp(
                    1.0 - (tree_k / torch.arange(TSRC - TDST, TSRC - TDST + q.shape[1], device=v.device)),
                    0.0,
                    1.0,
                )[None, :, None].to(v.dtype)
            )
            attn_output_hip = (attn_output_hip * (1 - scale_avg) + context_avg * scale_avg).to(v.dtype)

        attn_outputs.append(attn_output_hip)

        # Dense Queries Handling
        if LAST_DENSE_QUERIES is not None:
            flash_attention_mask = torch.zeros((N * H, abs(LAST_DENSE_QUERIES), TSRC), dtype=q.dtype, device=q.device)
            attn_output_last_flash, _ = flash_attention(q[:, LAST_DENSE_QUERIES:, :], k[:, :, :], v[:, :, :], flash_attention_mask)
            attn_outputs.append(attn_output_last_flash)

        if len(attn_outputs) > 1:
            attn_output = torch.cat(attn_outputs, dim=-2)
        else:
            attn_output = attn_outputs[0]

        attn_output = attn_output.view(N, H, TDST, HID)

    else:
        raise Exception(f"Attention method '{attention_method}' not fully implemented in this quick-fix script.")

    return attn_output, last_cumsum, attn_sparsity_loss
