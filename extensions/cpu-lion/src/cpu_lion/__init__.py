import torch  # Ensure libtorch shared libraries are loaded before importing the extension.

from .cpu_lion_interface import create_lion, destroy_lion, lion_is_avx512_enabled, lion_update
from .optimizer import CPULion

__all__ = [
    "CPULion",
    "lion_update",
    "create_lion",
    "destroy_lion",
    "lion_is_avx512_enabled",
]
