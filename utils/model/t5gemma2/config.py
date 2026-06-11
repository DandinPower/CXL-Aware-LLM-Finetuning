from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


_DEFAULT_THETA = {"global": 1_000_000.0, "local": 10_000.0}


@dataclass
class T5Gemma2TextConfig:
    vocab_size: int = 262_144
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    rms_norm_eps: float = 1e-6
    dropout_rate: float = 0.0
    attention_dropout: float = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 1024
    layer_types: list[str] | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    attention_bias: bool = False
    use_cache: bool = True
    use_bidirectional_attention: bool = False
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.layer_types is None:
            sliding_window_pattern = 6
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": _DEFAULT_THETA["local"]},
                "full_attention": {
                    "rope_type": "linear",
                    "factor": 8.0,
                    "rope_theta": _DEFAULT_THETA["global"],
                },
            }

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
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}.")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must match "
                f"num_hidden_layers ({self.num_hidden_layers})."
            )


@dataclass
class T5Gemma2EncoderConfig:
    text_config: T5Gemma2TextConfig = field(default_factory=T5Gemma2TextConfig)
    eoi_token_index: int = 256_000


@dataclass
class T5Gemma2DecoderConfig(T5Gemma2TextConfig):
    pass


@dataclass
class T5Gemma2Config:
    encoder: T5Gemma2EncoderConfig = field(default_factory=T5Gemma2EncoderConfig)
    decoder: T5Gemma2DecoderConfig = field(default_factory=T5Gemma2DecoderConfig)
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True
    image_token_index: int = 256_001
    eoi_token_index: int | None = None

    def __post_init__(self) -> None:
        if not self.is_encoder_decoder:
            raise ValueError("T5Gemma2Config requires is_encoder_decoder=True")

        if self.encoder.text_config.hidden_size != self.decoder.hidden_size:
            raise ValueError(
                "Imbalanced encoder-decoder hidden_size is unsupported: "
                f"{self.encoder.text_config.hidden_size} vs {self.decoder.hidden_size}."
            )

        if self.encoder.text_config.vocab_size != self.decoder.vocab_size:
            raise ValueError(
                "Imbalanced encoder-decoder vocab_size is unsupported: "
                f"{self.encoder.text_config.vocab_size} vs {self.decoder.vocab_size}."
            )

        if self.eoi_token_index is None:
            self.eoi_token_index = self.encoder.eoi_token_index

        # Keep a single source of truth for tied embedding behavior.
        self.encoder.text_config.tie_word_embeddings = self.tie_word_embeddings
        self.decoder.tie_word_embeddings = self.tie_word_embeddings

    @property
    def vocab_size(self) -> int:
        return self.decoder.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.decoder.hidden_size

    @classmethod
    def for_4b_4b(
        cls,
        vocab_size: int = 262_144,
        tie_word_embeddings: bool = True,
        official_config_path: str | Path | None = None,
    ) -> "T5Gemma2Config":
        if official_config_path is not None:
            loaded = cls.from_json_file(official_config_path)
            loaded.tie_word_embeddings = tie_word_embeddings
            loaded.encoder.text_config.tie_word_embeddings = tie_word_embeddings
            loaded.decoder.tie_word_embeddings = tie_word_embeddings
            return loaded

        text_4b = T5Gemma2TextConfig(
            vocab_size=vocab_size,
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=34,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            sliding_window=1024,
            rope_parameters={
                "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": _DEFAULT_THETA["global"]},
                "sliding_attention": {"rope_type": "default", "rope_theta": _DEFAULT_THETA["local"]},
            },
        )
        return cls(
            encoder=T5Gemma2EncoderConfig(text_config=text_4b, eoi_token_index=256_000),
            decoder=T5Gemma2DecoderConfig(**asdict(text_4b)),
            tie_word_embeddings=tie_word_embeddings,
            image_token_index=256_001,
        )

    @classmethod
    def for_tiny(
        cls,
        vocab_size: int = 1024,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 32,
    ) -> "T5Gemma2Config":
        text = T5Gemma2TextConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=4096,
            sliding_window=512,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": _DEFAULT_THETA["global"]},
                "sliding_attention": {"rope_type": "default", "rope_theta": _DEFAULT_THETA["local"]},
            },
        )
        return cls(
            encoder=T5Gemma2EncoderConfig(text_config=text, eoi_token_index=256_000),
            decoder=T5Gemma2DecoderConfig(**asdict(text)),
            tie_word_embeddings=True,
            image_token_index=256_001,
        )

    @classmethod
    def from_hf_config_dict(cls, cfg: dict[str, Any]) -> "T5Gemma2Config":
        encoder_dict = cfg.get("encoder", {})
        decoder_dict = cfg.get("decoder", {})
        text_dict = encoder_dict.get("text_config", {})

        # Decoder values are the main source of truth; fallback to encoder text config then top-level values.
        def pick(key: str, default: Any = None) -> Any:
            if key in decoder_dict:
                return decoder_dict[key]
            if key in text_dict:
                return text_dict[key]
            return cfg.get(key, default)

        layer_types = pick("layer_types")
        rope_parameters = pick("rope_parameters")

        shared_text = T5Gemma2TextConfig(
            vocab_size=int(pick("vocab_size", 262_144)),
            hidden_size=int(pick("hidden_size", 2560)),
            intermediate_size=int(pick("intermediate_size", 10240)),
            num_hidden_layers=int(pick("num_hidden_layers", 34)),
            num_attention_heads=int(pick("num_attention_heads", 8)),
            num_key_value_heads=int(pick("num_key_value_heads", 4)),
            head_dim=int(pick("head_dim", 256)),
            hidden_activation=str(pick("hidden_activation", "gelu_pytorch_tanh")),
            max_position_embeddings=int(pick("max_position_embeddings", 131_072)),
            rms_norm_eps=float(pick("rms_norm_eps", 1e-6)),
            dropout_rate=float(pick("dropout_rate", 0.0)),
            attention_dropout=float(pick("attention_dropout", 0.0)),
            query_pre_attn_scalar=int(pick("query_pre_attn_scalar", 256)),
            sliding_window=pick("sliding_window", 1024),
            layer_types=layer_types,
            rope_parameters=rope_parameters,
            final_logit_softcapping=pick("final_logit_softcapping"),
            attn_logit_softcapping=pick("attn_logit_softcapping"),
            attention_bias=bool(pick("attention_bias", False)),
            use_cache=bool(pick("use_cache", True)),
            use_bidirectional_attention=bool(pick("use_bidirectional_attention", False)),
            pad_token_id=int(pick("pad_token_id", 0)),
            eos_token_id=int(pick("eos_token_id", 1)),
            bos_token_id=int(pick("bos_token_id", 2)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
        )

        encoder = T5Gemma2EncoderConfig(
            text_config=shared_text,
            eoi_token_index=int(encoder_dict.get("eoi_token_index", cfg.get("eoi_token_index", 256_000))),
        )
        decoder = T5Gemma2DecoderConfig(**asdict(shared_text))

        return cls(
            encoder=encoder,
            decoder=decoder,
            is_encoder_decoder=bool(cfg.get("is_encoder_decoder", True)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),
            image_token_index=int(cfg.get("image_token_index", 256_001)),
            eoi_token_index=int(cfg.get("eoi_token_index", encoder.eoi_token_index)),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "T5Gemma2Config":
        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls.from_hf_config_dict(cfg)


__all__ = [
    "T5Gemma2TextConfig",
    "T5Gemma2EncoderConfig",
    "T5Gemma2DecoderConfig",
    "T5Gemma2Config",
]
