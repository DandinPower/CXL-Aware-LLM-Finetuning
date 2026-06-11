from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as torch_checkpoint

from .config import Lfm2MoeConfig


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


class MoeModelOutput(NamedTuple):
    last_hidden_state: torch.Tensor


class CausalLMOutput(NamedTuple):
    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None


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

            # Avoid FLASH/EFFICIENT constraints for explicit masks.
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


def _build_causal_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask [batch, seq], got {tuple(attention_mask.shape)}")

    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    key_mask = attention_mask.to(torch.bool)[:, None, None, :]
    query_mask = attention_mask.to(torch.bool)[:, None, :, None]
    return causal & key_mask & query_mask


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out hidden states for padding tokens.
    """
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError(
                "Short-conv path expects attention_mask as 2D [batch, seq]. "
                f"Got {tuple(attention_mask.shape)}"
            )
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)).to(dtype)

    return hidden_states


class Lfm2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Lfm2MoeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        rope_params = config.rope_parameters or {}
        rope_theta = float(rope_params.get("rope_theta", Lfm2MoeConfig.default_theta()))
        dim = config.head_dim

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Lfm2MoeMLP(nn.Module):
    def __init__(self, config: Lfm2MoeConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm2MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = F.silu

        nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.use_expert_bias = config.use_expert_bias

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Lfm2MoeExperts(config)
        if self.use_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = router_logits.sigmoid()
        if self.use_expert_bias:
            scores_for_routing = routing_weights + self.expert_bias
            _, selected_experts = torch.topk(scores_for_routing, k=self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, dim=1, index=selected_experts).type_as(router_logits)
        else:
            routing_weights, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)

        routing_weights = routing_weights * self.routed_scaling_factor
        return selected_experts, routing_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_reshaped)
        selected_experts, routing_weights = self.route_tokens_to_experts(router_logits)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Lfm2MoeAttention(nn.Module):
    """Multi-headed self-attention."""

    def __init__(self, config: Lfm2MoeConfig, layer_idx: int, use_flash_attn2: bool) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.use_flash_attn2 = use_flash_attn2

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_layernorm = Lfm2MoeRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm2MoeRMSNorm(self.head_dim, eps=config.norm_eps)

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

        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        is_causal = attention_mask is None
        attn_output = _run_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=0.0,
            training=self.training,
            use_flash_attn2=self.use_flash_attn2,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class Lfm2MoeShortConv(nn.Module):
    def __init__(self, config: Lfm2MoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = hidden_states.shape[1]

        x = apply_mask_to_padding_states(hidden_states, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        b, c, x = BCx.chunk(3, dim=-2)
        bx = b * x

        conv_out = self.conv(bx)[..., :seq_len]
        y = c * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y


class Lfm2MoeDecoderLayer(nn.Module):
    def __init__(self, config: Lfm2MoeConfig, layer_idx: int, use_flash_attn2: bool) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = Lfm2MoeAttention(config, layer_idx, use_flash_attn2=use_flash_attn2)
        else:
            self.conv = Lfm2MoeShortConv(config, layer_idx)

        self.feed_forward = (
            Lfm2MoeMLP(config, intermediate_size=config.intermediate_size)
            if layer_idx < config.num_dense_layers
            else Lfm2MoeSparseMoeBlock(config)
        )
        self.operator_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if self.is_attention_layer:
            if position_embeddings is None:
                raise ValueError("position_embeddings must be provided for attention layers")
            hidden_states = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                attention_mask=attention_mask,
            )

        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class Lfm2MoeModel(nn.Module):
    def __init__(self, config: Lfm2MoeConfig, use_flash_attn2: bool = True) -> None:
        super().__init__()
        self.config = config

        if use_flash_attn2 and _flash_attn_func is None:
            raise RuntimeError(
                "use_flash_attn2=True requires flash-attn's flash_attn_func, but it is not importable."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [Lfm2MoeDecoderLayer(config, layer_idx=i, use_flash_attn2=use_flash_attn2) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.pos_emb = Lfm2MoeRotaryEmbedding(config)
        self.embedding_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)

    def set_flash_attention_2_enabled(self, enabled: bool) -> None:
        if enabled and _flash_attn_func is None:
            raise RuntimeError(
                "set_flash_attention_2_enabled(True) requires flash-attn's flash_attn_func, "
                "but it is not importable."
            )

        for layer in self.layers:
            if layer.is_attention_layer:
                layer.self_attn.use_flash_attn2 = enabled

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor | torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
    ) -> MoeModelOutput:
        del use_cache  # no KV-cache support in this minimal pure-torch implementation

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = _embed_inputs(input_ids, self.embed_tokens, self.config.vocab_size)

        batch_size, seq_len, _ = inputs_embeds.shape

        if attention_mask is None:
            if input_ids is not None and input_ids.ndim == 2:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=inputs_embeds.device)
        else:
            if attention_mask.ndim != 2:
                raise ValueError(f"attention_mask must be 2D [batch, seq], got {tuple(attention_mask.shape)}")
            attention_mask = attention_mask.to(torch.bool)

        if position_ids is None:
            position_ids = _get_position_ids(batch_size, seq_len, inputs_embeds.device)

        hidden_states = inputs_embeds
        position_embeddings = self.pos_emb(hidden_states, position_ids=position_ids)

        causal_attention_mask = _build_causal_attention_mask(attention_mask)
        has_padding = not bool(attention_mask.all().item())

        for layer in self.layers:
            if layer.is_attention_layer:
                if layer.self_attn.use_flash_attn2:
                    if has_padding:
                        raise RuntimeError(
                            "use_flash_attn2=True does not support attention padding masks with "
                            "_flash_attn_func in this pure-torch Lfm2Moe path. "
                            "No fallback implementation is allowed."
                        )
                    layer_mask = None
                else:
                    layer_mask = causal_attention_mask
            else:
                layer_mask = attention_mask

            if self.gradient_checkpointing and self.training:
                hidden_states = torch_checkpoint.checkpoint(
                    lambda x, layer=layer, pos=position_embeddings, mask=layer_mask: layer(
                        x,
                        position_embeddings=pos,
                        attention_mask=mask,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                )

        hidden_states = self.embedding_norm(hidden_states)
        return MoeModelOutput(last_hidden_state=hidden_states)


class Lfm2MoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: Lfm2MoeConfig,
        enable_flash_attn2: bool = True,
        enable_liger_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2MoeModel(config, use_flash_attn2=enable_flash_attn2)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.use_liger_kernel = enable_liger_kernel
        if enable_liger_kernel and LigerFusedLinearCrossEntropyLoss is not None:
            self.liger_fused_ce = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                softcap=config.final_logit_softcapping,
            )
        else:
            self.liger_fused_ce = None

    def set_flash_attention_2_enabled(self, enabled: bool) -> None:
        self.model.set_flash_attention_2_enabled(enabled)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()

    def _compute_loss(
        self,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        use_liger_fused_lm_head: bool,
    ) -> torch.Tensor:
        # Causal LM shift: token t predicts token t+1.
        if hidden_states.shape[1] < 2:
            return hidden_states.sum() * 0.0

        hidden_for_loss = hidden_states[:, :-1, :]

        if labels.ndim == 3:
            if labels.shape[-1] != self.vocab_size:
                raise ValueError(
                    f"labels last dimension must equal vocab_size ({self.vocab_size}), got {labels.shape[-1]}."
                )

            shifted_labels = labels[:, 1:, :]

            if use_liger_fused_lm_head:
                targets = shifted_labels.argmax(dim=-1).to(torch.long)
                return self.liger_fused_ce(
                    self.lm_head.weight,
                    hidden_for_loss.reshape(-1, hidden_for_loss.size(-1)),
                    targets.reshape(-1),
                )

            if logits is None:
                raise RuntimeError("logits must be computed when fused Liger lm_head path is disabled.")
            shifted_logits = logits[:, :-1, :]
            log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
            loss = -(shifted_labels.float() * log_probs).sum(dim=-1).mean()
            return loss

        if labels.ndim != 2:
            raise ValueError("labels must be 2D token ids or 3D distributions")

        targets = labels[:, 1:].to(torch.long)

        if use_liger_fused_lm_head:
            return self.liger_fused_ce(
                self.lm_head.weight,
                hidden_for_loss.reshape(-1, hidden_for_loss.size(-1)),
                targets.reshape(-1),
            )

        if logits is None:
            raise RuntimeError("logits must be computed when fused Liger lm_head path is disabled.")
        shifted_logits = logits[:, :-1, :]
        return F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> CausalLMOutput:
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        hidden_states = model_outputs.last_hidden_state
        use_liger_fused_lm_head = (
            self.training and labels is not None and self.liger_fused_ce is not None and hidden_states.is_cuda
        )

        logits = None
        if not use_liger_fused_lm_head:
            logits_hidden = hidden_states
            if labels is None:
                if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                    logits_hidden = hidden_states[:, -logits_to_keep:, :]
                elif not isinstance(logits_to_keep, int):
                    logits_hidden = hidden_states[:, logits_to_keep, :]

            logits = self.lm_head(logits_hidden)

            if self.config.final_logit_softcapping is not None:
                cap = self.config.final_logit_softcapping
                logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            loss = self._compute_loss(
                logits=logits,
                hidden_states=hidden_states,
                labels=labels,
                use_liger_fused_lm_head=use_liger_fused_lm_head,
            )

        return CausalLMOutput(logits=logits, loss=loss)


def flash_attn2_extension_available() -> bool:
    return _flash_attn_func is not None


def liger_kernel_available() -> bool:
    return LigerFusedLinearCrossEntropyLoss is not None


__all__ = [
    "MoeModelOutput",
    "CausalLMOutput",
    "flash_attn2_extension_available",
    "liger_kernel_available",
    "Lfm2MoeModel",
    "Lfm2MoeForCausalLM",
]
