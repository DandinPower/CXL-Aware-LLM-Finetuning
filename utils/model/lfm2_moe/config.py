from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class Lfm2MoeConfig:
    model_type: str = "lfm2_moe"

    vocab_size: int = 65_536
    hidden_size: int = 2048
    intermediate_size: int = 7168
    moe_intermediate_size: int = 1792
    num_hidden_layers: int = 32

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int | list[int] = 2

    tie_word_embeddings: bool = True
    rope_parameters: dict[str, Any] | None = None
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    use_cache: bool = True
    norm_eps: float = 1e-5

    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    conv_bias: bool = False
    conv_L_cache: int = 3

    num_dense_layers: int = 2
    num_experts_per_tok: int = 4
    num_experts: int = 32
    use_expert_bias: bool = True
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True

    full_attn_idxs: list[int] | None = None
    layer_types: list[str] | None = None

    # Optional compatibility field for fused loss behavior.
    final_logit_softcapping: float | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def default_theta(cls) -> float:
        return 1_000_000.0

    def __post_init__(self) -> None:
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.default_theta(),
            }

        if self.layer_types is None:
            if self.full_attn_idxs is None:
                self.full_attn_idxs = list(range(self.num_hidden_layers))
            full_attn_set = set(self.full_attn_idxs)
            self.layer_types = [
                "full_attention" if i in full_attn_set else "conv" for i in range(self.num_hidden_layers)
            ]

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        if self.num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be > 0, got {self.num_experts_per_tok}.")

        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) cannot exceed "
                f"num_experts ({self.num_experts})."
            )

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must match "
                f"num_hidden_layers ({self.num_hidden_layers})."
            )

        if self.num_dense_layers < 0 or self.num_dense_layers > self.num_hidden_layers:
            raise ValueError(
                f"num_dense_layers must be within [0, num_hidden_layers], got {self.num_dense_layers}."
            )

    @classmethod
    def for_tiny(
        cls,
        vocab_size: int = 1024,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        moe_intermediate_size: int = 64,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        num_dense_layers: int = 1,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
    ) -> "Lfm2MoeConfig":
        # Alternate attention/conv to cover both operator paths in tiny tests.
        layer_types = ["full_attention" if i % 2 == 0 else "conv" for i in range(num_hidden_layers)]
        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_dense_layers=num_dense_layers,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            layer_types=layer_types,
            max_position_embeddings=4096,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": cls.default_theta(),
            },
        )

    @classmethod
    def for_8b_a1b(
        cls,
        vocab_size: int = 65_536,
        tie_word_embeddings: bool = True,
        official_config_path: str | Path | None = None,
    ) -> "Lfm2MoeConfig":
        if official_config_path is not None:
            loaded = cls.from_json_file(official_config_path)
            loaded.tie_word_embeddings = tie_word_embeddings
            return loaded

        # Matches LiquidAI/LFM2-8B-A1B architecture defaults.
        return cls(
            vocab_size=vocab_size,
            hidden_size=2048,
            intermediate_size=7168,
            moe_intermediate_size=1792,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=8,
            conv_bias=False,
            conv_L_cache=3,
            num_dense_layers=2,
            num_experts_per_tok=4,
            num_experts=32,
            use_expert_bias=True,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            max_position_embeddings=128_000,
            norm_eps=1e-5,
            tie_word_embeddings=tie_word_embeddings,
            layer_types=[
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
            ],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": cls.default_theta(),
            },
        )

    @classmethod
    def from_hf_config_dict(cls, cfg: dict[str, Any]) -> "Lfm2MoeConfig":
        data = dict(cfg)

        # Keep compatibility with aliases that may appear in upstream checkpoints.
        if "tie_embedding" in data and "tie_word_embeddings" not in data:
            data["tie_word_embeddings"] = bool(data["tie_embedding"])
        if "block_ff_dim" in data and "intermediate_size" not in data:
            data["intermediate_size"] = int(data["block_ff_dim"])
        if "rope_theta" in data and "rope_parameters" not in data:
            data["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": float(data["rope_theta"]),
            }

        supported_keys = set(cls.__dataclass_fields__.keys())
        init_kwargs = {k: v for k, v in data.items() if k in supported_keys}
        return cls(**init_kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Lfm2MoeConfig":
        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls.from_hf_config_dict(cfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["Lfm2MoeConfig"]
