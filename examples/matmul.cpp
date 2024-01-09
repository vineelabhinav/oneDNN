/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void matmul_example(dnnl::engine::kind engine_kind) {

// auto tg=tag::abc;

// Create execution dnnl::engine.
dnnl::engine engine(engine_kind, 0);

// Create dnnl::stream.
dnnl::stream engine_stream(engine);

// Tensor dimensions.
const memory::dim MB = 1, // batch size
            M = 1, K = 1, N =1; //N=512 only zeros

auto tg=tag::ab;
memory::dims src_dims = {M,K};
memory::dims weights_dims = {K,N};
memory::dims dst_dims = {M,N};

// M=20;
// K=18;
// N=503;
// auto tg=tag::abc;
// memory::dims src_dims = {MB, M, K};
// memory::dims weights_dims = {MB, K, N};
// memory::dims dst_dims = {MB, M, N};


// Source (src), weights, bias, and destination (dst) tensors dimensions.
// memory::dims src_dims = {1,12,128,128};
// memory::dims weights_dims = { 1,12,128,64};

// memory::dims dst_dims = {1,12,128,64};

// Allocate buffers.
std::vector<float> src_data(product(src_dims));
std::vector<float> weights_data(product(weights_dims));
// std::vector<float> bias_data(product(bias_dims));
std::vector<float> dst_data(product(dst_dims));

// std::generate(src_data.begin(), src_data.end(), []() {
//         static int i = 0;
//         return std::cos(i++ / 10.f);
//     });
//     std::generate(weights_data.begin(), weights_data.end(), []() {
//         static int i = 0;
//         return std::sin(i++ * 2.f);
//     });



std::generate(src_data.begin(), src_data.end(), []() {
static int i = 1;
return i++;
});
std::generate(weights_data.begin(), weights_data.end(), []() {
static int i = 1;
// i++;
return i++;
});
//for (int i=0;i<weights_data.size();i++)
//        {
//                std::cout<<weights_data[i]<<" ";
//        }
//        std::cout<<"\n";
// std::generate(bias_data.begin(), bias_data.end(), []() {
//     static int i = 0;
//     return std::tanh(float(i++));
// });

// Create memory descriptors and memory objects for src, weights, bias, and
// dst.
auto src_md = memory::desc(src_dims, dt::f32, tg);
auto weights_md = memory::desc(weights_dims, dt::f32, tg);
// auto bias_md = memory::desc(bias_dims, dt::f32, tg);
auto dst_md = memory::desc(dst_dims, dt::f32, tg);

auto src_mem = memory(src_md, engine);
auto weights_mem = memory(weights_md, engine);
// auto bias_mem = memory(bias_md, engine);
auto dst_mem = memory(dst_md, engine);
std::cout<<"input: \n";
    for (int i=0;i<src_data.size();i++)
{
    std::cout<<src_data[i]<<" ";
}
std::cout<<"\n";
    for (int i=0;i<weights_data.size();i++)
{
    std::cout<<weights_data[i]<<" ";
}
std::cout<<"\n";
 
// Write data to memory object's handles.
write_to_dnnl_memory(src_data.data(), src_mem);
write_to_dnnl_memory(weights_data.data(), weights_mem);
// write_to_dnnl_memory(bias_data.data(), bias_mem);

// Create primitive post-ops (ReLU).
const float alpha = 0.f;
const float beta = 0.f;
post_ops matmul_ops;
matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
primitive_attr matmul_attr;
matmul_attr.set_post_ops(matmul_ops);

// Create primitive descriptor.
auto matmul_pd = matmul::primitive_desc(
engine, src_md, weights_md,  dst_md, matmul_attr);

// Create the primitive.
auto matmul_prim = matmul(matmul_pd);

// Primitive arguments.
std::unordered_map<int, memory> matmul_args;
matmul_args.insert({DNNL_ARG_SRC, src_mem});
matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
// matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
matmul_args.insert({DNNL_ARG_DST, dst_mem});

// Primitive execution: matrix multiplication with ReLU.
matmul_prim.execute(engine_stream, matmul_args);

// Wait for the computation to finalize.
// engine_stream.wait();

// Read data from memory object's handle.
std::cout<<"output : "<<dst_data.size() <<"\n";
read_from_dnnl_memory(dst_data.data(), dst_mem);
// dst_data.size()
for (int i=0;i<dst_data.size();i++)
{
std::cout<<dst_data[i]<<" ";
}
std::cout<<"\n";
}

int main(int argc, char **argv) {
return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}


// onednn_verbose,exec,cpu,matmul,gemm:jit:f32,undef,src_f32::blocked:ab::f0 wei_f32::blocked:ab::f0 dst_f32::blocked:ab::f0,attr-post-ops:eltwise_relu ,,4x2:2x5,0.849854
// 13 16 19 22 25 27 34 41 48 55 41 52 63 74 85 55 70 85 100 115 Example passed on CPU.


// onednn_verbose,exec,cpu,matmul,brg:sve_256,undef,src_f32::blocked:ab::f0 wei_f32::blocked:ab::f0 dst_f32::blocked:ab::f0,attr-post-ops:eltwise_relu ,,4x2:2x5,0.0461426
// 13 16 19 22 25 27 34 41 48 55 41 52 63 74 85 55 70 85 100 115 Example passed on CPU.
