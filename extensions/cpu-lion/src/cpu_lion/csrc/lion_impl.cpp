// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "lion.h"

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_1(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size, _params, grads, _exp_avg, _param_size);
#endif
    if (_param_size > rounded_size) {
        const float beta1_minus1 = 1.0f - _betta1;
        const float beta2_minus1 = 1.0f - _betta2;
        const float neg_lr = -1.0f * _alpha;
        const float wd_scale = 1.0f - (_alpha * _weight_decay);

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = static_cast<float>(grads[k]);
                float param = static_cast<float>(_params[k]);
                float momentum = static_cast<float>(_exp_avg[k]);

                if (_weight_decay > 0.0f) { param *= wd_scale; }

                float update = (momentum * _betta1) + (grad * beta1_minus1);
                update = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);
                param += neg_lr * update;

                momentum = (momentum * _betta2) + (grad * beta2_minus1);

                _params[k] = static_cast<ds_params_precision_t>(param);
                _exp_avg[k] = static_cast<ds_state_precision_t>(momentum);
            }
        }
    }
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_4(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size, _params, grads, _exp_avg, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_param_size - rounded_size));
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void Lion_Optimizer::Step_8(ds_params_precision_t* _params,
                            ds_params_precision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size, _params, grads, _exp_avg, _param_size);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_param_size - rounded_size));
}

int create_lion_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float weight_decay,
                          bool should_log)
{
#if !defined(__AVX512__)
    throw std::runtime_error("Lion CPU optimizer requires an AVX512 build and runtime support.");
#endif

    auto opt = std::make_shared<Lion_Optimizer>(alpha, betta1, betta2, weight_decay);
    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        printf("Lion Optimizer #%d is created with AVX512 arithmetic capability.\n", optimizer_id);
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f\n",
               alpha,
               betta1,
               betta2,
               weight_decay);
    }

    return 0;
}

template <typename ds_params_precision_t, typename ds_state_precision_t>
void step_invoker(std::shared_ptr<Lion_Optimizer> opt,
                  void* _params,
                  void* grads,
                  void* _exp_avg,
                  size_t _param_size)
{
    opt->Step_8((ds_params_precision_t*)(_params),
                (ds_params_precision_t*)(grads),
                (ds_state_precision_t*)(_exp_avg),
                _param_size);
}

std::map<std::tuple<c10::ScalarType, c10::ScalarType>,
         std::function<void(std::shared_ptr<Lion_Optimizer>, void*, void*, void*, size_t)>>
    invokers;

template <class ds_params_precision_t, class ds_state_precision_t>
void create_invoker()
{
    invokers[std::tuple(c10::CppTypeToScalarType<ds_params_precision_t>(),
                        c10::CppTypeToScalarType<ds_state_precision_t>())] =
        step_invoker<ds_params_precision_t, ds_state_precision_t>;
}

struct InvokerInitializer {
    InvokerInitializer()
    {
        create_invoker<c10::Half, float>();
        create_invoker<c10::Half, c10::Half>();
        create_invoker<c10::BFloat16, float>();
        create_invoker<c10::BFloat16, c10::BFloat16>();
        create_invoker<float, float>();
    }
} _invoker_initializer;

void invoke(std::shared_ptr<Lion_Optimizer> opt,
            torch::Tensor& params,
            torch::Tensor& grads,
            torch::Tensor& exp_avg,
            size_t param_size)
{
    c10::ScalarType params_type = at::typeMetaToScalarType(params.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

    auto it = invokers.find(std::tuple(params_type, state_type));
    if (it == invokers.end()) {
        throw std::runtime_error("Lion optimizer with param type "s + c10::toString(params_type) +
                                 " and state type "s + c10::toString(state_type) +
                                 " is not supported on current hardware"s);
    }

    it->second(opt, params.data_ptr(), grads.data_ptr(), exp_avg.data_ptr(), param_size);
}

int ds_lion_step(int optimizer_id,
                 float lr,
                 float beta1,
                 float beta2,
                 float weight_decay,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg)
{
    TORCH_CHECK(params.device().is_cpu(), "params must be a CPU tensor");
    TORCH_CHECK(grads.device().is_cpu(), "grads must be a CPU tensor");
    TORCH_CHECK(exp_avg.device().is_cpu(), "exp_avg must be a CPU tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
    TORCH_CHECK(params.numel() == grads.numel(), "params and grads must have the same number of elements");
    TORCH_CHECK(params.numel() == exp_avg.numel(),
                "params and exp_avg must have the same number of elements");

    auto opt_it = s_optimizers.find(optimizer_id);
    TORCH_CHECK(opt_it != s_optimizers.end(), "No lion optimizer found for optimizer_id=", optimizer_id);
    std::shared_ptr<Lion_Optimizer> opt = std::static_pointer_cast<Lion_Optimizer>(opt_it->second);
    opt->update_state(lr, beta1, beta2, weight_decay);

    invoke(opt, params, grads, exp_avg, params.numel());

    return 0;
}

int destroy_lion_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);
    return 0;
}

bool lion_is_avx512_enabled()
{
#if defined(__AVX512__)
    return true;
#else
    return false;
#endif
}
