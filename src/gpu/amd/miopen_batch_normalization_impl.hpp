/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_AMD_MIOPEN_BATCH_NORMALIZATION_IMPL_HPP
#define GPU_AMD_MIOPEN_BATCH_NORMALIZATION_IMPL_HPP

#include "gpu/amd/sycl_hip_utils.hpp"
#include <miopen/miopen.h>

#define MIOPEN_BN_MIN_EPSILON 1e-5

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct bnorm_args_t {
public:
    bnorm_args_t(void *x, void *mean, void *var, void *scale, void *bias)
        : x_(x), mean_(mean), var_(var), scale_(scale), bias_(bias) {}

    void *x_, *mean_, *var_, *scale_, *bias_;
};

struct bnorm_fwd_args_t : public bnorm_args_t {
    bnorm_fwd_args_t(void *x, void *y, void *mean, void *var, void *scale,
            void *bias, void *y_prime, void *save_mean, void *save_var)
        : bnorm_args_t::bnorm_args_t(x, mean, var, scale, bias)
        , y_(y)
        , y_prime_(y_prime)
        , save_mean_(save_mean)
        , save_var_(save_var) {}

    void *y_, *y_prime_, *save_mean_, *save_var_;
};

struct bnorm_bwd_args_t : public bnorm_args_t {
    bnorm_bwd_args_t(void *x, void *dx, void *dy, void *mean, void *var,
            void *scale, void *diff_scale, void *diff_bias, void *wkspace,
            void *relu_dx)
        : bnorm_args_t(x, mean, var, scale, nullptr)
        , dx_(dx)
        , dy_(dy)
        , diff_scale_(diff_scale)
        , diff_bias_(diff_bias)
        , wkspace_(wkspace)
        , relu_dx_(relu_dx) {}

    void *dx_, *dy_, *diff_scale_, *diff_bias_, *wkspace_, *relu_dx_;
};

struct miopen_batch_normalization_impl_base_t {
    virtual ~miopen_batch_normalization_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }

        if ((fuse_norm_relu_ || with_relu_postop_) && act_desc_) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyActivationDescriptor, act_desc_);
        }
    }

    virtual status_t init(batch_normalization_pd_t *pd) = 0;

    virtual void execute(miopenHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const = 0;

    bool is_bwd_d() const { return is_bwd_data_; }
    bool is_training() const { return is_training_; }
    bool use_global_stats() const { return use_global_stats_; }
    bool use_scale() const { return use_scale_; }
    bool use_shift() const { return use_shift_; }
    bool fuse_norm_relu() const { return fuse_norm_relu_; }
    std::size_t dt_size() const { return dt_size_; }
    std::size_t mean_var_size_bytes() { return mean_var_size_bytes_; }
    uint8_t default_mean_var() const { return 0; }
    int C() const { return nchannels_; }

protected:
    status_t init_common(batch_normalization_pd_t *pd) {
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
        if (ndims_ > 5) { return status::invalid_arguments; }

        memory_desc_wrapper wrap(pd->src_md());
        use_scale_ = pd->use_scale();
        use_shift_ = pd->use_shift();
        fuse_norm_relu_ = pd->fuse_norm_relu();
        is_training_ = pd->is_training();
        use_global_stats_ = pd->use_global_stats();
        is_bwd_data_ = pd->desc()->prop_kind == prop_kind::backward_data;
        dt_size_ = types::data_type_size(wrap.data_type());
        nchannels_ = pd->C();
        // Mean and variance are always f32.
        mean_var_size_bytes_
                = nchannels_ * types::data_type_size(data_type::f32);
        eps_ = pd->desc()->batch_norm_epsilon;
        y_prime_size_ = wrap.nelems() * types::data_type_size(data_type::f32);
        with_relu_postop_ = pd->with_relu_post_op();

        auto n = static_cast<float>(pd->MB() * pd->D() * pd->H() * pd->W());
        var_scaling_factor_ = (n - 1.f) / n;

        convert_dims(pd->src_md()->padded_dims, dims_[src], pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_[src],
                pd->ndims());

        CHECK(convert_data_type(pd->src_md(), &data_types_[src]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src],
                data_types_[src], ndims_, dims_[src], strides_[src]));
        CHECK(create_and_set_scaleshift_desc());

        if (fuse_norm_relu_ || with_relu_postop_) {
            CHECK(create_and_set_activation_desc());
        }

        return status::success;
    }

    virtual status_t create_and_set_scaleshift_desc() {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateTensorDescriptor, &tensor_descs_[scl]));

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenDeriveBNTensorDescriptor,
                tensor_descs_[scl], tensor_descs_[src], mode_));

        return status::success;
    }

    virtual status_t create_and_set_activation_desc() {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &act_desc_));

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, act_desc_,
                miopenActivationRELU, activAlpha, activBeta, activGamma));

        return status::success;
    }

    virtual status_t to_population_variance(
            miopenHandle_t handle, void *var) const {
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenScaleTensor, handle,
                tensor_descs_[scl], var, &var_scaling_factor_));
        return status::success;
    }

    enum io { src = 0, dst, scl, NUM_IO };
    miopenDataType_t data_types_[NUM_IO];
    miopenTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    miopenActivationDescriptor_t act_desc_;
    miopenBatchNormMode_t mode_ = miopenBNSpatial;
    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    int strides_[NUM_IO][DNNL_MAX_NDIMS];
    int ndims_, nchannels_;
    float alpha_ = 1.f, beta = 0.f;
    double relu_coef_ = 0.0;
    double factor_ = 1.0;
    double eps_ = MIOPEN_BN_MIN_EPSILON;
    float var_scaling_factor_ = 0.f;
    bool use_scale_ = false;
    bool use_shift_ = false;
    bool fuse_norm_relu_ = false;
    bool with_relu_postop_ = false;
    bool use_global_stats_ = false;
    bool is_training_ = false;
    bool is_bwd_data_ = false;
    double activAlpha = 0.0;
    double activBeta;
    double activGamma;
    std::size_t y_prime_size_;
    std::size_t dt_size_, mean_var_size_bytes_;
};

struct miopen_batch_normalization_fwd_impl_t
    : public miopen_batch_normalization_impl_base_t {
    using miopen_batch_normalization_impl_base_t::
            miopen_batch_normalization_impl_base_t;

    status_t init(batch_normalization_pd_t *pd) override {
        init_common(pd);

        convert_dims(pd->dst_md()->padded_dims, dims_[dst], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides_[dst],
                pd->ndims());

        CHECK(convert_data_type(pd->dst_md(), &data_types_[dst]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                data_types_[dst], ndims_, dims_[dst], strides_[dst]));

        return status::success;
    }

    void execute(miopenHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        auto fwd_args = static_cast<bnorm_fwd_args_t *>(args.get());

        MIOPEN_EXECUTE_FUNC(miopenBatchNormalizationForwardTraining, handle,
                mode_, (void *)&alpha_, (void *)&beta, tensor_descs_[src],
                fwd_args->x_, tensor_descs_[dst], fwd_args->y_,
                tensor_descs_[scl], fwd_args->scale_, fwd_args->bias_, factor_,
                fwd_args->mean_, fwd_args->var_, eps_, fwd_args->save_mean_,
                fwd_args->save_var_);

        if (is_training_) { to_population_variance(handle, fwd_args->var_); }

        if (fuse_norm_relu_ || with_relu_postop_) { do_relu(handle, fwd_args); }
    }

protected:
    void do_relu(miopenHandle_t handle, bnorm_fwd_args_t *fwd_args) const {
        if (is_training_ && fuse_norm_relu_) {
            // Copy the result to the workspace
            int alpha2 = 0;
            miopenTensorOp_t tensorOp = miopenTensorOpAdd;

            MIOPEN_EXECUTE_FUNC(miopenOpTensor, handle, tensorOp, &alpha_,
                    tensor_descs_[dst], fwd_args->y_, &alpha2,
                    tensor_descs_[dst], fwd_args->y_, &beta, tensor_descs_[dst],
                    fwd_args->y_prime_);
        }
        MIOPEN_EXECUTE_FUNC(miopenActivationForward, handle, act_desc_, &alpha_,
                tensor_descs_[dst], fwd_args->y_, &beta, tensor_descs_[dst],
                fwd_args->y_);
    }
};

struct miopen_batch_normalization_fwd_stats_impl_t
    : public miopen_batch_normalization_fwd_impl_t {

    status_t init(batch_normalization_pd_t *pd) override {
        return miopen_batch_normalization_fwd_impl_t::init(pd);
    }

    void execute(miopenHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        auto fwd_args = static_cast<bnorm_fwd_args_t *>(args.get());

        MIOPEN_EXECUTE_FUNC(miopenBatchNormalizationForwardInference, handle,
                mode_, (void *)&alpha_, (void *)&beta, tensor_descs_[src],
                fwd_args->x_, tensor_descs_[dst], fwd_args->y_,
                tensor_descs_[scl], fwd_args->scale_, fwd_args->bias_,
                fwd_args->mean_, fwd_args->var_, eps_);

        if (fuse_norm_relu_ || with_relu_postop_) { do_relu(handle, fwd_args); }
    }
};

struct miopen_batch_normalization_bwd_impl_t
    : public miopen_batch_normalization_impl_base_t {

    status_t init(batch_normalization_pd_t *pd) override {
        init_common(pd);

        convert_dims(pd->diff_src_md()->padded_dims, diff_dims_[diff_src],
                pd->ndims());
        convert_dims(pd->diff_dst_md()->padded_dims, diff_dims_[diff_dst],
                pd->ndims());
        convert_dims(pd->diff_src_md()->format_desc.blocking.strides,
                strides_[diff_src], pd->ndims());
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides_[diff_dst], pd->ndims());

        CHECK(convert_data_type(
                pd->diff_src_md(), &diff_data_types_[diff_src]));
        CHECK(convert_data_type(
                pd->diff_dst_md(), &diff_data_types_[diff_dst]));

        CHECK(create_and_set_tensor_descriptor(&diff_tensor_descs_[diff_src],
                diff_data_types_[diff_src], ndims_, diff_dims_[diff_src],
                strides_[diff_src]));
        CHECK(create_and_set_tensor_descriptor(&diff_tensor_descs_[diff_dst],
                diff_data_types_[diff_dst], ndims_, diff_dims_[diff_dst],
                strides_[diff_dst]));

        return status::success;
    }

    void execute(miopenHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        auto bwd_args = static_cast<bnorm_bwd_args_t *>(args.get());

        MIOPEN_EXECUTE_FUNC(miopenBatchNormalizationBackward, handle, mode_,
                &a_data_diff_, &b_data_diff_, &a_param_diff_, &b_param_diff_,
                tensor_descs_[src], bwd_args->x_, diff_tensor_descs_[diff_dst],
                bwd_args->dy_, diff_tensor_descs_[diff_src], bwd_args->dx_,
                tensor_descs_[scl], bwd_args->scale_, bwd_args->diff_scale_,
                bwd_args->diff_bias_, eps_, bwd_args->mean_, bwd_args->var_);
    }

    ~miopen_batch_normalization_bwd_impl_t() {
        for (size_t i = 0; i < NUM_DIFF; i++) {
            if (diff_tensor_descs_[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, diff_tensor_descs_[i]);
            }
        }
    }

protected:
    const float a_data_diff_ = 1.f, b_data_diff_ = 0.f;
    const float a_param_diff_ = 1.f, b_param_diff_ = 0.f;

    enum diff_tensors { diff_src = 0, diff_dst, NUM_DIFF };
    int diff_dims_[NUM_DIFF][DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t diff_tensor_descs_[NUM_DIFF] = {};
    miopenDataType_t diff_data_types_[NUM_DIFF];
};

struct miopen_batch_normalization_bwd_relu_impl_t
    : public miopen_batch_normalization_bwd_impl_t {

    status_t init(batch_normalization_pd_t *pd) override {
        pd->scratchpad_registry().registrar().book(
                memory_tracking::names::key_none,
                memory_desc_wrapper(pd->diff_dst_md()).size(), size_t(1));

        return miopen_batch_normalization_bwd_impl_t::init(pd);
    }

    void execute(miopenHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        auto bwd_args = static_cast<bnorm_bwd_args_t *>(args.get());

        MIOPEN_EXECUTE_FUNC(miopenActivationBackward, handle, act_desc_,
                &alpha_, diff_tensor_descs_[dst], bwd_args->wkspace_,
                diff_tensor_descs_[dst], bwd_args->dy_, diff_tensor_descs_[dst],
                bwd_args->wkspace_, &beta, diff_tensor_descs_[dst],
                bwd_args->relu_dx_);

        MIOPEN_EXECUTE_FUNC(miopenBatchNormalizationBackward, handle, mode_,
                &a_data_diff_, &b_data_diff_, &a_param_diff_, &b_param_diff_,
                tensor_descs_[src], bwd_args->x_, diff_tensor_descs_[dst],
                bwd_args->relu_dx_, diff_tensor_descs_[src], bwd_args->dx_,
                tensor_descs_[scl], bwd_args->scale_, bwd_args->diff_scale_,
                bwd_args->diff_bias_, eps_, bwd_args->mean_, bwd_args->var_);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
