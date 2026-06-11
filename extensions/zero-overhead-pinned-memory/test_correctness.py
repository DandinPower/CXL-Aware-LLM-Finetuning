import pytest
import torch
import random

from zero_overhead_pinned_memory import to_posix_memalign_pinned, to_cuda_host_alloc_pinned, zeros_cuda_host_alloc_pinned

@pytest.mark.parametrize("shape", [
    (10,),
    (128, 128),
    (64, 64, 64),
])
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16
])
def test_to_posix_memalign_pinned(shape, dtype):
    """Test that to_posix_memalign_pinned produces a pinned CPU tensor
    with the same shape & values as the original."""
    random.seed(42)
    torch.manual_seed(42)
    original = torch.randn(shape, dtype=dtype)
    pinned = to_posix_memalign_pinned(original)
    assert pinned.device.type == 'cpu', "Tensor is not on CPU."
    assert pinned.is_pinned(), "Returned tensor is not marked as pinned."
    assert pinned.shape == original.shape, "Shape mismatch."
    assert pinned.dtype == original.dtype, "Dtype mismatch."
    if pinned.is_floating_point():
        assert torch.allclose(pinned, original), "Pinned tensor values differ from original (float)."
    else:
        assert torch.equal(pinned, original), "Pinned tensor values differ from original (int)."
    if torch.cuda.is_available():
        try:
            gpu_tensor = pinned.to('cuda')
            assert torch.allclose(gpu_tensor.cpu(), original), "GPU copy does not match original values."
        except Exception as e:
            pytest.fail(f"Transferring pinned tensor to GPU raised an exception: {e}")


@pytest.mark.parametrize("shape", [
    (10,),
    (128, 128),
    (64, 64, 64),
])
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16
])
def test_to_cuda_host_alloc_pinned(shape, dtype):
    """Test that to_cuda_host_alloc_pinned produces a pinned CPU tensor
    with the same shape & values as the original."""
    random.seed(42)
    torch.manual_seed(42)
    original = torch.randn(shape, dtype=dtype)
    pinned = to_cuda_host_alloc_pinned(original)
    assert pinned.device.type == 'cpu', "Tensor is not on CPU."
    assert pinned.is_pinned(), "Returned tensor is not marked as pinned."
    assert pinned.shape == original.shape, "Shape mismatch."
    assert pinned.dtype == original.dtype, "Dtype mismatch."
    if pinned.is_floating_point():
        assert torch.allclose(pinned, original), "Pinned tensor values differ from original (float)."
    else:
        assert torch.equal(pinned, original), "Pinned tensor values differ from original (int)."
    if torch.cuda.is_available():
        try:
            gpu_tensor = pinned.to('cuda')
            assert torch.allclose(gpu_tensor.cpu(), original), "GPU copy does not match original values."
        except Exception as e:
            pytest.fail(f"Transferring pinned tensor to GPU raised an exception: {e}")

@pytest.mark.parametrize("shape", [
    (10,),
    (128, 128),
    (64, 64, 64),
])
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16
])
def test_zeros_cuda_host_alloc_pinned(shape, dtype):
    """Test that zeros_cuda_host_alloc_pinned produces a pinned CPU tensor,
    filled with zeros, with the specified shape & dtype, and aligned to 4096 bytes."""
    pinned = zeros_cuda_host_alloc_pinned(shape, dtype)
    assert pinned.device.type == 'cpu', "Tensor is not on CPU."
    assert pinned.is_pinned(), "Returned tensor is not marked as pinned."
    assert pinned.shape == shape, "Shape mismatch."
    assert pinned.dtype == dtype, "Dtype mismatch."
    assert torch.allclose(pinned, torch.zeros(shape, dtype=dtype)), "Pinned tensor is not all zeros."
    DMA_ALIGNMENT = 4096
    assert pinned.data_ptr() % DMA_ALIGNMENT == 0, "Data pointer is not aligned to 4096 bytes."
    if torch.cuda.is_available():
        try:
            gpu_tensor = pinned.to('cuda')
            assert torch.allclose(gpu_tensor.cpu(), pinned), "GPU copy does not match original values."
        except Exception as e:
            pytest.fail(f"Transferring pinned tensor to GPU raised an exception: {e}")