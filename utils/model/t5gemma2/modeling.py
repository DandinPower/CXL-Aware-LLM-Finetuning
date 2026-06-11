from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as torch_checkpoint

from .config import T5Gemma2Config, T5Gemma2DecoderConfig, T5Gemma2TextConfig


try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
    except Exception:
        _flash_attn_func = None


try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
except Exception:
    LigerFusedLinearCrossEntropyLoss = None


class Seq2SeqModelOutput(NamedTuple):
    last_hidden_state: torch.Tensor
    encoder_last_hidden_state: torch.Tensor


class Seq2SeqLMOutput(NamedTuple):
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
    dropout_p: float,
    training: bool,
    use_flash_attn2: bool,
) -> torch.Tensor:
    if use_flash_attn2:
        if _flash_attn_func is None:
            raise RuntimeError(
                "use_flash_attn2=True requires flash-attn's flash_attn_func, but it is not importable."
            )
        if not q.is_cuda:
            raise RuntimeError("use_flash_attn2=True requires CUDA tensors for Q/K/V.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"use_flash_attn2=True requires fp16/bf16 Q/K/V tensors, got dtype={q.dtype}."
            )
        if attention_mask is not None:
            raise RuntimeError(
                "use_flash_attn2=True does not allow explicit attention_mask in this path. "
                "No fallback implementation is allowed."
            )

        # flash-attn2 expects [batch, seq, heads, head_dim]
        q_f = q.transpose(1, 2)
        k_f = k.transpose(1, 2)
        v_f = v.transpose(1, 2)
        try:
            out = _flash_attn_func(
                q_f,
                k_f,
                v_f,
                dropout_p=dropout_p if training else 0.0,
                causal=is_causal,
            )
            return out.transpose(1, 2)
        except Exception as exc:
            raise RuntimeError(
                "use_flash_attn2=True requested strict flash-attn2 execution, "
                "but _flash_attn_func failed and fallback is disabled."
            ) from exc

    if attention_mask is not None and q.is_cuda:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # Avoid FLASH/EFFICIENT constraints for explicit masks (e.g., stride/alignment requirements).
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p if training else 0.0,
                    is_causal=is_causal,
                )
        except Exception:
            pass

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=is_causal,
    )


def _embed_inputs(
    input_ids: torch.Tensor,
    embed_tokens: nn.Embedding,
    vocab_size: int,
) -> torch.Tensor:
    if input_ids.ndim == 2:
        return embed_tokens(input_ids)

    if input_ids.ndim == 3 and input_ids.size(-1) == vocab_size:
        return input_ids.to(embed_tokens.weight.dtype) @ embed_tokens.weight

    raise ValueError(
        "Expected input_ids shape [batch, seq] (token ids) or [batch, seq, vocab_size] (token distribution). "
        f"Got {tuple(input_ids.shape)}."
    )


def _get_position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.LongTensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def _build_encoder_attention_mask(attention_mask: torch.Tensor | None, q_len: int) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    if attention_mask.ndim == 2:
        mask = attention_mask.to(torch.bool)[:, None, None, :]
        return mask.expand(-1, 1, q_len, -1)

    if attention_mask.ndim == 4:
        return attention_mask.to(torch.bool)

    raise ValueError(f"Unsupported attention_mask shape for encoder: {tuple(attention_mask.shape)}")


def _build_merged_attention_mask(
    batch_size: int,
    q_len: int,
    enc_len: int,
    device: torch.device,
    decoder_attention_mask: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    causal = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=device))
    self_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, q_len)

    if decoder_attention_mask is not None:
        dec_key_mask = decoder_attention_mask.to(torch.bool)[:, None, None, :]
        self_mask = self_mask & dec_key_mask

    if encoder_attention_mask is None:
        cross_mask = torch.ones((batch_size, 1, q_len, enc_len), dtype=torch.bool, device=device)
    else:
        if encoder_attention_mask.ndim != 2:
            raise ValueError(
                "encoder_attention_mask for this minimal implementation must have shape [batch, enc_seq_len]."
            )
        cross_mask = encoder_attention_mask.to(torch.bool)[:, None, None, :].expand(-1, 1, q_len, -1)

    return torch.cat([self_mask, cross_mask], dim=-1)


class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(dtype=x.dtype)


class T5Gemma2MLP(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        return self.down_proj(hidden_states)


class T5Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.inv_freq_by_layer_type: dict[str, torch.Tensor] = {}

        for layer_type in set(config.layer_types):
            rope_cfg = config.rope_parameters.get(layer_type, {})
            theta = float(rope_cfg.get("rope_theta", 1_000_000.0))
            rope_type = str(rope_cfg.get("rope_type", "default"))
            rope_factor = float(rope_cfg.get("factor", 1.0))
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim))
            )
            if rope_type == "linear" and rope_factor > 0.0:
                inv_freq = inv_freq / rope_factor
            self.register_buffer(f"inv_freq_{layer_type}", inv_freq, persistent=False)
            self.inv_freq_by_layer_type[layer_type] = inv_freq

    def forward(
        self,
        position_ids: torch.LongTensor,
        layer_type: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq_by_layer_type[layer_type].to(device=device)
        freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin


class T5Gemma2SelfAttention(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int, use_flash_attn2: bool) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.use_flash_attn2 = use_flash_attn2
        self.query_scale = (float(config.head_dim) / float(config.query_pre_attn_scalar)) ** 0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = T5Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = T5Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q * self.query_scale

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_output = _run_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            is_causal=False,
            dropout_p=self.attention_dropout,
            training=self.training,
            use_flash_attn2=self.use_flash_attn2,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class T5Gemma2MergedAttention(nn.Module):
    def __init__(self, config: T5Gemma2DecoderConfig, layer_idx: int, use_flash_attn2: bool) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.use_flash_attn2 = use_flash_attn2
        self.query_scale = (float(config.head_dim) / float(config.query_pre_attn_scalar)) ** 0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = T5Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = T5Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, dec_len, _ = hidden_states.shape
        _, enc_len, _ = encoder_hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, dec_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        self_k = self.k_proj(hidden_states).view(batch_size, dec_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        self_v = self.v_proj(hidden_states).view(batch_size, dec_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cross_k = self.k_proj(encoder_hidden_states).view(batch_size, enc_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cross_v = self.v_proj(encoder_hidden_states).view(batch_size, enc_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        self_k = self.k_norm(self_k)
        cross_k = self.k_norm(cross_k)
        q = q * self.query_scale

        cos, sin = position_embeddings
        q, self_k = apply_rotary_pos_emb(q, self_k, cos, sin)

        self_k = repeat_kv(self_k, self.num_key_value_groups)
        self_v = repeat_kv(self_v, self.num_key_value_groups)
        cross_k = repeat_kv(cross_k, self.num_key_value_groups)
        cross_v = repeat_kv(cross_v, self.num_key_value_groups)

        attn_mask = merged_attention_mask
        is_causal = False
        if self.use_flash_attn2:
            mask = merged_attention_mask.to(torch.bool)
            if mask.ndim != 4 or mask.shape != (batch_size, 1, dec_len, dec_len + enc_len):
                raise RuntimeError(
                    "use_flash_attn2=True requires merged_attention_mask with shape "
                    f"[batch, 1, dec_len, dec_len + enc_len], got {tuple(mask.shape)}."
                )

            cross_part = mask[..., dec_len:]
            if not bool(cross_part.all().item()):
                raise RuntimeError(
                    "use_flash_attn2=True does not support encoder padding in merged attention "
                    "with _flash_attn_func. No fallback implementation is allowed."
                )

            self_part = mask[..., :dec_len]
            expected_causal = torch.tril(torch.ones((dec_len, dec_len), dtype=torch.bool, device=mask.device))
            if not bool(torch.equal(self_part, expected_causal.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1))):
                raise RuntimeError(
                    "use_flash_attn2=True does not support decoder padding in merged attention "
                    "with _flash_attn_func. No fallback implementation is allowed."
                )

            # With causal=True and seqlen_k > seqlen_q, flash-attn2 uses bottom-right alignment.
            # Ordering keys as [cross, self] yields: all cross keys + causal self keys.
            k = torch.cat([cross_k, self_k], dim=2)
            v = torch.cat([cross_v, self_v], dim=2)
            attn_mask = None
            is_causal = True
        else:
            k = torch.cat([self_k, cross_k], dim=2)
            v = torch.cat([self_v, cross_v], dim=2)

        attn_output = _run_attention(
            q,
            k,
            v,
            attention_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attention_dropout,
            training=self.training,
            use_flash_attn2=self.use_flash_attn2,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, dec_len, -1)
        return self.o_proj(attn_output)


class T5Gemma2EncoderLayer(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int, use_flash_attn2: bool) -> None:
        super().__init__()
        self.self_attn = T5Gemma2SelfAttention(config, layer_idx, use_flash_attn2=use_flash_attn2)
        self.pre_self_attn_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = T5Gemma2MLP(config)
        self.pre_feedforward_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class T5Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: T5Gemma2DecoderConfig, layer_idx: int, use_flash_attn2: bool) -> None:
        super().__init__()
        self.self_attn = T5Gemma2MergedAttention(config, layer_idx, use_flash_attn2=use_flash_attn2)
        self.pre_self_attn_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = T5Gemma2MLP(config)
        self.pre_feedforward_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        merged_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            merged_attention_mask=merged_attention_mask,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


class T5Gemma2TextEncoder(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig, shared_embeddings: nn.Embedding, use_flash_attn2: bool) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = shared_embeddings
        self.layers = nn.ModuleList(
            [T5Gemma2EncoderLayer(config, layer_idx=i, use_flash_attn2=use_flash_attn2) for i in range(config.num_hidden_layers)]
        )
        self.norm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(config)
        self.gradient_checkpointing = False

    def set_flash_attention_2_enabled(self, enabled: bool) -> None:
        for layer in self.layers:
            layer.self_attn.use_flash_attn2 = enabled

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = _embed_inputs(input_ids, self.embed_tokens, self.config.vocab_size)

        batch_size, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            position_ids = _get_position_ids(batch_size, seq_len, inputs_embeds.device)

        if attention_mask is None and input_ids is not None and input_ids.ndim == 2:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        if attention_mask is not None:
            attention_mask = _build_encoder_attention_mask(attention_mask, q_len=seq_len)
            if self.layers and self.layers[0].self_attn.use_flash_attn2:
                if bool(attention_mask.all().item()):
                    attention_mask = None
                else:
                    raise RuntimeError(
                        "use_flash_attn2=True does not support encoder padding masks with _flash_attn_func. "
                        "No fallback implementation is allowed."
                    )

        hidden_states = self.dropout(inputs_embeds)

        position_embeddings = {
            layer_type: self.rotary_emb(
                position_ids=position_ids,
                layer_type=layer_type,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            for layer_type in set(self.config.layer_types)
        }

        for i, layer in enumerate(self.layers):
            layer_position_embeddings = position_embeddings[self.config.layer_types[i]]

            if self.gradient_checkpointing and self.training:
                hidden_states = torch_checkpoint.checkpoint(
                    lambda x, layer=layer, layer_position_embeddings=layer_position_embeddings, attention_mask=attention_mask: layer(
                        x, layer_position_embeddings, attention_mask
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, layer_position_embeddings, attention_mask)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class T5Gemma2Decoder(nn.Module):
    def __init__(self, config: T5Gemma2DecoderConfig, shared_embeddings: nn.Embedding, use_flash_attn2: bool) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = shared_embeddings
        self.layers = nn.ModuleList(
            [T5Gemma2DecoderLayer(config, layer_idx=i, use_flash_attn2=use_flash_attn2) for i in range(config.num_hidden_layers)]
        )
        self.norm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(config)
        self.gradient_checkpointing = False

    def set_flash_attention_2_enabled(self, enabled: bool) -> None:
        for layer in self.layers:
            layer.self_attn.use_flash_attn2 = enabled

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided to decoder")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of decoder input_ids or decoder inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = _embed_inputs(input_ids, self.embed_tokens, self.config.vocab_size)

        batch_size, dec_len, _ = inputs_embeds.shape
        enc_len = encoder_hidden_states.shape[1]

        if position_ids is None:
            position_ids = _get_position_ids(batch_size, dec_len, inputs_embeds.device)

        if attention_mask is None:
            if input_ids is not None and input_ids.ndim == 2:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = torch.ones((batch_size, dec_len), dtype=torch.bool, device=inputs_embeds.device)

        merged_attention_mask = _build_merged_attention_mask(
            batch_size=batch_size,
            q_len=dec_len,
            enc_len=enc_len,
            device=inputs_embeds.device,
            decoder_attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = self.dropout(inputs_embeds)

        position_embeddings = {
            layer_type: self.rotary_emb(
                position_ids=position_ids,
                layer_type=layer_type,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            for layer_type in set(self.config.layer_types)
        }

        for i, layer in enumerate(self.layers):
            layer_position_embeddings = position_embeddings[self.config.layer_types[i]]

            if self.gradient_checkpointing and self.training:
                hidden_states = torch_checkpoint.checkpoint(
                    lambda x, enc, layer=layer, layer_position_embeddings=layer_position_embeddings, merged_attention_mask=merged_attention_mask: layer(
                        x, enc, layer_position_embeddings, merged_attention_mask
                    ),
                    hidden_states,
                    encoder_hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, encoder_hidden_states, layer_position_embeddings, merged_attention_mask)

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class T5Gemma2Model(nn.Module):
    def __init__(self, config: T5Gemma2Config, use_flash_attn2: bool = True) -> None:
        super().__init__()
        self.config = config
        if use_flash_attn2 and _flash_attn_func is None:
            raise RuntimeError(
                "use_flash_attn2=True requires flash-attn's flash_attn_func, but it is not importable."
            )
        self.shared = nn.Embedding(
            config.decoder.vocab_size,
            config.decoder.hidden_size,
            padding_idx=config.decoder.pad_token_id,
        )
        self.encoder = T5Gemma2TextEncoder(config.encoder.text_config, self.shared, use_flash_attn2=use_flash_attn2)
        self.decoder = T5Gemma2Decoder(config.decoder, self.shared, use_flash_attn2=use_flash_attn2)

    def set_flash_attention_2_enabled(self, enabled: bool) -> None:
        if enabled and _flash_attn_func is None:
            raise RuntimeError(
                "set_flash_attention_2_enabled(True) requires flash-attn's flash_attn_func, "
                "but it is not importable."
            )
        self.encoder.set_flash_attention_2_enabled(enabled)
        self.decoder.set_flash_attention_2_enabled(enabled)

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
    ) -> Seq2SeqModelOutput:
        if encoder_outputs is None:
            encoder_hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        else:
            encoder_hidden_states = encoder_outputs

        decoder_hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_hidden_states,
            encoder_last_hidden_state=encoder_hidden_states,
        )


class T5Gemma2ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: T5Gemma2Config,
        enable_flash_attn2: bool = True,
        enable_liger_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = T5Gemma2Model(config, use_flash_attn2=enable_flash_attn2)
        self.vocab_size = config.decoder.vocab_size
        self.lm_head = nn.Linear(config.decoder.hidden_size, self.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.shared.weight

        self.use_liger_kernel = enable_liger_kernel
        if enable_liger_kernel and LigerFusedLinearCrossEntropyLoss is not None:
            self.liger_fused_ce = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                softcap=config.decoder.final_logit_softcapping,
            )
        else:
            self.liger_fused_ce = None

    def set_flash_attention_2_enabled(self, enabled: bool) -> None:
        self.model.set_flash_attention_2_enabled(enabled)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.LongTensor:
        if labels.ndim == 3:
            labels = labels.argmax(dim=-1)
        elif labels.ndim != 2:
            raise ValueError(
                "labels must be shaped [batch, seq] (token ids) or [batch, seq, vocab_size] (distribution)."
            )

        decoder_start_token_id = self.config.decoder.bos_token_id
        pad_token_id = self.config.decoder.pad_token_id

        shifted = labels.new_full(labels.shape, pad_token_id)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted.masked_fill_(shifted.eq(-100), pad_token_id)
        return shifted

    def _compute_loss(
        self,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        use_liger_fused_lm_head: bool,
    ) -> torch.Tensor:
        if labels.ndim == 3:
            if labels.shape[-1] != self.vocab_size:
                raise ValueError(
                    f"labels last dimension must equal vocab_size ({self.vocab_size}), got {labels.shape[-1]}."
                )

            if use_liger_fused_lm_head:
                targets = labels.argmax(dim=-1).to(torch.long)
                return self.liger_fused_ce(
                    self.lm_head.weight,
                    hidden_states.reshape(-1, hidden_states.size(-1)),
                    targets.reshape(-1),
                )

            if logits is None:
                raise RuntimeError("logits must be computed when fused Liger lm_head path is disabled.")
            log_probs = F.log_softmax(logits.float(), dim=-1)
            loss = -(labels.float() * log_probs).sum(dim=-1).mean()
            return loss

        if labels.ndim != 2:
            raise ValueError("labels must be 2D token ids or 3D distributions")

        targets = labels.to(torch.long)

        if use_liger_fused_lm_head:
            return self.liger_fused_ce(
                self.lm_head.weight,
                hidden_states.reshape(-1, hidden_states.size(-1)),
                targets.reshape(-1),
            )

        if logits is None:
            raise RuntimeError("logits must be computed when fused Liger lm_head path is disabled.")
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        encoder_outputs: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        decoder_inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )

        hidden_states = model_outputs.last_hidden_state
        use_liger_fused_lm_head = (
            self.training and labels is not None and self.liger_fused_ce is not None and hidden_states.is_cuda
        )

        logits = None
        if not use_liger_fused_lm_head:
            logits = self.lm_head(hidden_states)

            if self.config.decoder.final_logit_softcapping is not None:
                cap = self.config.decoder.final_logit_softcapping
                logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            loss = self._compute_loss(
                logits=logits,
                hidden_states=hidden_states,
                labels=labels,
                use_liger_fused_lm_head=use_liger_fused_lm_head,
            )

        return Seq2SeqLMOutput(
            logits=logits,
            loss=loss,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state,
        )


def flash_attn2_extension_available() -> bool:
    return _flash_attn_func is not None


def liger_kernel_available() -> bool:
    return LigerFusedLinearCrossEntropyLoss is not None


__all__ = [
    "Seq2SeqModelOutput",
    "Seq2SeqLMOutput",
    "flash_attn2_extension_available",
    "liger_kernel_available",
    "T5Gemma2ForConditionalGeneration",
    "T5Gemma2Model",
]
