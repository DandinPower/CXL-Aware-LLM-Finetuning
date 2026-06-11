#pragma once

#include <torch/extension.h>
#include <cassert>
#include <cmath>
#include "simd.h"

#define STEP(SPAN)                                                           \
    template <typename ds_params_precision_t, typename ds_state_precision_t> \
    void Step_##SPAN(ds_params_precision_t* _params,                         \
                     ds_params_precision_t* grads,                           \
                     ds_state_precision_t* _exp_avg,                         \
                     size_t _param_size);

class Lion_Optimizer {
public:
    Lion_Optimizer(float alpha = 1e-4, float betta1 = 0.9, float betta2 = 0.99, float weight_decay = 0)
        : _alpha(alpha), _betta1(betta1), _betta2(betta2), _weight_decay(weight_decay)
    {
    }
    ~Lion_Optimizer() {}

#if defined(__AVX512__) or defined(__AVX256__)
    template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
    void Step_AVX(size_t* rounded_size,
                  ds_params_precision_t* _params,
                  ds_params_precision_t* grads,
                  ds_state_precision_t* _exp_avg,
                  size_t param_size);
#endif

    STEP(1)
    STEP(4)
    STEP(8)

    inline void update_state(float lr, float beta1, float beta2, float weight_decay)
    {
        _alpha = lr;
        _betta1 = beta1;
        _betta2 = beta2;
        _weight_decay = weight_decay;
    }

private:
    float _alpha;
    float _betta1;
    float _betta2;
    float _weight_decay;
};

#if defined(__AVX512__) or defined(__AVX256__)
template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_AVX(size_t* rounded_size,
                              ds_params_precision_t* _params,
                              ds_params_precision_t* grads,
                              ds_state_precision_t* _exp_avg,
                              size_t _param_size)
{
#if !defined(__AVX512__)
    if (std::is_same_v<ds_params_precision_t, c10::BFloat16> ||
        std::is_same_v<ds_state_precision_t, c10::BFloat16>) {
        return;
    }
#endif
    size_t new_rounded_size = 0;

    AVX_Data beta1_4;
    beta1_4.data = SIMD_SET(_betta1);
    AVX_Data beta2_4;
    beta2_4.data = SIMD_SET(_betta2);

    AVX_Data beta1_minus1_4;
    beta1_minus1_4.data = SIMD_SET(1.0f - _betta1);
    AVX_Data beta2_minus1_4;
    beta2_minus1_4.data = SIMD_SET(1.0f - _betta2);

    AVX_Data neg_lr_4;
    neg_lr_4.data = SIMD_SET(-1.0f * _alpha);

    AVX_Data wd_scale_4;
    wd_scale_4.data = SIMD_SET(1.0f - (_alpha * _weight_decay));

    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            AVX_Data grad_4[span];
            simd_load<span>(grad_4, grads + i);

            AVX_Data momentum_4[span];
            simd_load<span>(momentum_4, _exp_avg + i);

            AVX_Data param_4[span];
            simd_load<span>(param_4, _params + i);

            if (_weight_decay > 0.0f) { simd_mul<span>(param_4, param_4, wd_scale_4); }

            AVX_Data update_4[span];
            simd_mul<span>(update_4, momentum_4, beta1_4);
            simd_fma<span>(update_4, grad_4, beta1_minus1_4, update_4);
            simd_sign<span>(update_4, update_4);

            simd_fma<span>(param_4, update_4, neg_lr_4, param_4);
            simd_store<span>(_params + i, param_4);

            simd_mul<span>(momentum_4, momentum_4, beta2_4);
            simd_fma<span>(momentum_4, grad_4, beta2_minus1_4, momentum_4);
            simd_store<span>(_exp_avg + i, momentum_4);
        }
    }
    *rounded_size = new_rounded_size;
}
#endif

int create_lion_optimizer(int optimizer_id,
                          float alpha = 1e-4,
                          float betta1 = 0.9,
                          float betta2 = 0.99,
                          float weight_decay = 0.0,
                          bool should_log = false);

int ds_lion_step(int optimizer_id,
                 float lr,
                 float beta1,
                 float beta2,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg);

int destroy_lion_optimizer(int optimizer_id);

bool lion_is_avx512_enabled();
