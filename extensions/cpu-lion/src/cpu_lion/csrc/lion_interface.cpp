// SPDX-License-Identifier: Apache-2.0

#include "lion.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lion_update", &ds_lion_step, "Lion CPU update (C++)");
    m.def("create_lion", &create_lion_optimizer, "Lion CPU create (C++)");
    m.def("destroy_lion", &destroy_lion_optimizer, "Lion CPU destroy (C++)");
    m.def("lion_is_avx512_enabled", &lion_is_avx512_enabled, "Lion AVX512 build flag");
}
