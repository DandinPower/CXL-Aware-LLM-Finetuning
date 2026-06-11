import torch

def main():
    assert torch.cuda.is_available(), "CUDA is not available. flash-attn requires a CUDA GPU."

    try:
        from flash_attn import flash_attn_func
    except ImportError as e:
        raise ImportError(
            "Failed to import flash-attn. Make sure it is installed correctly:\n"
            "  pip install flash-attn --no-build-isolation"
        ) from e

    device = "cuda"
    dtype = torch.float16

    batch_size = 2
    seq_len = 128
    num_heads = 4
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

    print("flash-attn import: OK")
    print("flash-attn forward pass: OK")
    print("Output shape:", out.shape)
    print("Output dtype:", out.dtype)
    print("Output device:", out.device)

    expected_shape = (batch_size, seq_len, num_heads, head_dim)
    assert out.shape == expected_shape, f"Unexpected output shape: {out.shape}"

    if torch.isnan(out).any():
        raise RuntimeError("Output contains NaNs.")

    print("Verification passed.")


if __name__ == "__main__":
    main()