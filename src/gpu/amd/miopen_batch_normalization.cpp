/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/amd/miopen_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

status_t miopen_batch_normalization_fwd_t::execute(
        const exec_ctx_t &ctx) const {
    return miopen_batch_normalization_common_t::execute(
            ctx, ctx.stream()->engine(), pd());
}

status_t miopen_batch_normalization_bwd_t::execute(
        const exec_ctx_t &ctx) const {
    return miopen_batch_normalization_common_t::execute(
            ctx, ctx.stream()->engine(), pd());
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
