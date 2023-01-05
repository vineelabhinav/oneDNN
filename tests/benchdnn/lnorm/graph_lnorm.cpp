/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "lnorm/graph_lnorm.hpp"

#include <string>
#include <vector>

namespace benchdnnext {
namespace lnorm {

static std::vector<int64_t> get_stat_dims(
        const std::vector<int64_t> &lnorm_dims) {
    return std::vector<int64_t>(
            lnorm_dims.begin(), lnorm_dims.begin() + (lnorm_dims.size() - 1));
}

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::lnorm::prb_t *prb) {
    std::vector<dnnl::graph::logical_tensor::data_type> dts;
    for (const auto dt : prb->dt) {
        dts.push_back(convert_dt(dt));
    }
    return dts;
}

static int check_known_skipped_case_graph(
        const ::lnorm::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::lnorm::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    check_known_skipped_case_graph_common(
            {prb->dt}, normalize_tag(prb->tag[0], prb->ndims), prb->dir, res);
    if (res->state == SKIPPED) return OK;
    /* GLOBAL STATS cannot be passed as DNNL Graph doesnt support this */
    if (prb->flags & ::lnorm::GLOB_STATS) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    /* STAT_TAG=tag::undef not supported */
    if (prb->stat_tag == tag::undef) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    /* oneDNN Graph supports either both scale and shift or neither of them.
     In order to run the test with the use_affine=true attribute,
     use dnnl_use_scale and dnnl_use_shift flag (--flags=CH). */
    if (prb->use_sc() != prb->use_sh()) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    return OK;
}

static quant_data_t get_qdata_for(const ::lnorm::prb_t *prb) {
    const auto q_dt = convert_dt(prb->dt[1]);
    const auto src_scales = prb->attr.scales.get(DNNL_ARG_SRC);
    const auto dst_scales = prb->attr.scales.get(DNNL_ARG_DST);
    const auto scale = dst_scales.scale / src_scales.scale;
    //scale is set to the quant's attr with reciprocal for each scale in graph side
    std::vector<float> scales = {scale};
    const std::vector<int64_t> zps(scales.size(), 0L);
    const std::string q_type = "per_tensor";
    return quant_data_t(q_dt, scales, zps, q_type, 0, prb->tag[1]);
}

fill_status_t append_graph_with_block(const ::lnorm::prb_t *prb) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;
    using graph_lt = dnnl::graph::logical_tensor::layout_type;
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto with_tc = with_typecast(orig_dts)
            || with_typecast_after(orig_dts[0], orig_dts[1]);

    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::LNORM);
    const auto src_lt_kind
            = prb->dir == FLAG_FWD ? lt_kind::SRC : lt_kind::DIFF_SRC;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);
    const auto sc_id = graph.generate_id_for(op_id, lt_kind::SC);
    const auto d_sc_id = graph.generate_id_for(op_id, lt_kind::DIFF_SC);
    const auto sh_id = graph.generate_id_for(op_id, lt_kind::SH);
    const auto d_sh_id = graph.generate_id_for(op_id, lt_kind::DIFF_SH);
    const auto mean_id = graph.generate_id_for(op_id, lt_kind::MEAN);
    const auto var_id = graph.generate_id_for(op_id, lt_kind::VAR);

    const auto common_dt = convert_dt(prb->dt[0]);
    dims_t base_dims = prb->dims;
    dims_t stat_dims = get_stat_dims(prb->dims);
    dims_t ss_dims = {prb->c};

    graph.create_lt(src_id, common_dt, base_dims, graph_lt::strided);
    graph.create_lt(dst_id, common_dt, base_dims, graph_lt::strided);
    if (prb->dir == FWD_D || prb->dir & FLAG_BWD) {
        graph.create_lt(mean_id, graph_dt::f32, stat_dims, graph_lt::strided);
        graph.create_lt(var_id, graph_dt::f32, stat_dims, graph_lt::strided);
    }
    if (prb->use_sc() && prb->use_sh()) {
        graph.create_lt(sc_id, graph_dt::f32, ss_dims, graph_lt::strided);
        graph.create_lt(sh_id, graph_dt::f32, ss_dims, graph_lt::strided);
    }
    if (prb->dir & FLAG_BWD && prb->use_sc() && prb->use_sh()) {
        graph.create_lt(d_sc_id, graph_dt::f32, ss_dims, graph_lt::strided);
        graph.create_lt(d_sh_id, graph_dt::f32, ss_dims, graph_lt::strided);
    }

    std::vector<size_t> src_ids;
    std::vector<size_t> dst_ids;
    dnnl::graph::op::kind lnorm_kind;
    if (prb->dir & FLAG_FWD) {
        src_ids = {src_id};
        dst_ids = {dst_id};
        if (prb->dir == FWD_D) {
            dst_ids.push_back(mean_id);
            dst_ids.push_back(var_id);
        }
        lnorm_kind = dnnl::graph::op::kind::LayerNorm;
    } else {
        const auto fwd_src_id = graph.generate_id_for(op_id, lt_kind::SRC);
        graph.create_lt(fwd_src_id, common_dt, base_dims, graph_lt::strided);
        src_ids = {fwd_src_id, dst_id, mean_id, var_id};
        dst_ids = {src_id};
        if (prb->use_sc() && prb->use_sh()) {
            dst_ids.push_back(d_sc_id);
            dst_ids.push_back(d_sh_id);
        }
        lnorm_kind = dnnl::graph::op::kind::LayerNormBackprop;
    }
    if (prb->use_sc() && prb->use_sh()) {
        src_ids.push_back(sc_id);
        src_ids.push_back(sh_id);
    }

    dnnl::graph::op lnorm_op(op_id, lnorm_kind, graph.stringify_id(op_id));
    lnorm_op.set_attr("begin_norm_axis", int64_t(-1))
            .set_attr("use_affine", prb->use_sc() && prb->use_sh())
            .set_attr("epsilon", float(1.f / 16));
    if (prb->dir & FLAG_FWD) lnorm_op.set_attr("keep_stats", prb->dir == FWD_D);

    graph.append(op_id, lnorm_op, src_ids, dst_ids);

    fill_status_t status;
    // add typecast op after lnorm
    if (with_tc) {
        //The typecast in lnorm's fusion pattern only support bf16<-->f32.
        graph_dt dst_dt = orig_dts[0] == graph_dt::bf16 ? graph_dt::f32
                                                        : graph_dt::bf16;
        status = insert_typecast_after(graph.get_cur_block_out_id(), dst_dt);
        BENCHDNNEXT_VERIFY(status);
    }

    // if required - add quantize op
    if (is_low_precision({orig_dts[1]})) {
        status = insert_quant_after(
                graph.get_cur_block_out_id(), get_qdata_for(prb));
        BENCHDNNEXT_VERIFY(status);
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::lnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    // Filter partitions
    const auto partitions = graph.get_partitions();
    if (partitions.empty() || partitions.size() > 1) {
        cleanup();
        return res->state = FAILED, FAIL;
    }

    const auto par = partitions[0];
    if (!par.is_supported()) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::lnorm::init_pd, prb, res, par, ins, outs);

    static const engine_t cpu_engine(dnnl_cpu);

    const bool keep_stats {prb->dir == FWD_D};
    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[0], (prb->tag[0]).c_str());
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t placeholder_dst_dt;
    if (!prb->inplace) {
        placeholder_dst_dt = make_dnn_mem(outs[0], (prb->tag[0]).c_str());
    }
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : placeholder_dst_dt;

    dnn_mem_t mean_fp, var_fp, mean_dt, var_dt;
    if (keep_stats || prb->dir & FLAG_BWD) {
        mean_dt = make_dnn_mem(prb->dir & FLAG_FWD ? outs[1] : ins[2],
                (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(
                prb->dir & FLAG_FWD ? outs[1] : ins[2], dt::f32, tag::abx);
        var_dt = make_dnn_mem(prb->dir & FLAG_FWD ? outs[2] : ins[3],
                (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(
                prb->dir & FLAG_FWD ? outs[2] : ins[3], dt::f32, tag::abx);
    } else {
        /* prepare_fwd needs these memories for inference and training cases.
         * Below memories are used when keep_stats=false */
        dnnl::graph::logical_tensor mean_lt(1,
                dnnl::graph::logical_tensor::data_type::f32,
                get_stat_dims(prb->dims), lt::strided);
        dnnl::graph::logical_tensor var_lt(2,
                dnnl::graph::logical_tensor::data_type::f32,
                get_stat_dims(prb->dims), lt::strided);
        mean_dt = make_dnn_mem(mean_lt, (prb->stat_tag).c_str());
        mean_fp = make_dnn_mem(mean_lt, dt::f32, tag::abx);
        var_dt = make_dnn_mem(var_lt, (prb->stat_tag).c_str());
        var_fp = make_dnn_mem(var_lt, dt::f32, tag::abx);
    }

    // this logical_tensor is used to indicate the memory size for scale
    dnnl::graph::logical_tensor lt_sc {3, dt::f32, dims_t {prb->c},
            dnnl::graph::logical_tensor::layout_type::strided};
    auto sc_fp = make_dnn_mem(lt_sc, dt::f32, tag::abx);
    auto sc_dt = make_dnn_mem(lt_sc, dt::f32, tag::abx);

    // this logical_tensor is used to indicate the memory size for shift
    dnnl::graph::logical_tensor lt_sh {4, dt::f32, dims_t {prb->c},
            dnnl::graph::logical_tensor::layout_type::strided};
    auto sh_fp = make_dnn_mem(lt_sh, dt::f32, tag::x);
    auto sh_dt = make_dnn_mem(lt_sh, dt::f32, tag::x);

    dnnl::graph::logical_tensor lt_scales {5, dt::f32, dims_t {1},
            dnnl::graph::logical_tensor::layout_type::strided};
    auto src_scales_fp = make_dnn_mem(lt_scales, dt::f32, tag::x);
    auto src_scales_dt = make_dnn_mem(lt_scales, dt::f32, tag::x);
    auto dst_scales_fp = make_dnn_mem(lt_scales, dt::f32, tag::x);
    auto dst_scales_dt = make_dnn_mem(lt_scales, dt::f32, tag::x);
    src_scales_fp.set_elem(0, 1);
    SAFE(src_scales_dt.reorder(src_scales_fp), WARN);
    dst_scales_fp.set_elem(0, prb->attr.scales.get(DNNL_ARG_DST).scale);
    SAFE(dst_scales_dt.reorder(dst_scales_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;

    if (prb->dir & FLAG_FWD) {
        if (::lnorm::prepare_fwd(prb, src_fp, mean_fp, var_fp, sc_fp, sh_fp)
                != OK) {
            cleanup();
            return res->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        //TODO - not supported for now.
        //TODO: not tested - as global stats not there in DNNL Graph
        if (prb->flags & ::lnorm::GLOB_STATS) {
            /* prepare mean & var if they are inputs */
            SAFE(mean_dt.reorder(mean_fp), WARN);
            SAFE(var_dt.reorder(var_fp), WARN);
        }

        // oneDNN Graph supports either both scale and shift or neither of them
        if (prb->use_sc() && prb->use_sh()) {
            SAFE(sc_dt.reorder(sc_fp), WARN);
            SAFE(sh_dt.reorder(sh_fp), WARN);
        }

        const dnnl::graph::engine &eng = get_test_engine();

        tensors_in.emplace_back(
                dnnl::graph::tensor(ins[0], eng, static_cast<void *>(src_dt)));
        tensors_out.emplace_back(
                dnnl::graph::tensor(outs[0], eng, static_cast<void *>(dst_dt)));
        auto gamma_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto beta_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        if (prb->use_sc() && prb->use_sh()) {
            for (int64_t i = 0; i < prb->c; i++) {
                gamma_v.set_elem(i, sc_dt.get_elem(i));
                beta_v.set_elem(i, sh_dt.get_elem(i));
            }
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[1], eng, static_cast<void *>(gamma_v)));
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins[2], eng, static_cast<void *>(beta_v)));
        }

        if (keep_stats) {
            tensors_out.emplace_back(dnnl::graph::tensor(
                    outs[1], eng, static_cast<void *>(mean_dt)));
            tensors_out.emplace_back(dnnl::graph::tensor(
                    outs[2], eng, static_cast<void *>(var_dt)));
        }

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args;
            args.set(DNNL_ARG_DST, dst_dt);

            args_t ref_args;
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE, sc_fp);
            ref_args.set(DNNL_ARG_SHIFT, sh_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_fp);
            ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_fp);

            std::vector<data_kind_t> kinds {DST};
            if (!(prb->flags & ::lnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                args.set(DNNL_ARG_MEAN, mean_dt);
                args.set(DNNL_ARG_VARIANCE, var_dt);
                kinds.push_back(MEAN);
                kinds.push_back(VAR);
            }

            check_correctness(
                    prb, kinds, args, ref_args, ::lnorm::setup_cmp, res);
        }
    } else {
        auto d_sc_fp = make_dnn_mem(lt_sc, dt::f32, tag::abx);
        auto d_sc_dt = make_dnn_mem(lt_sc, dt::f32, tag::abx);

        auto d_sh_fp = make_dnn_mem(lt_sh, dt::f32, tag::abx);
        auto d_sh_dt = make_dnn_mem(lt_sh, dt::f32, tag::abx);

        dnn_mem_t placeholder_d_src_dt;
        // backward pass
        auto d_dst_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        auto d_dst_dt = make_dnn_mem(ins[1], (prb->tag[0]).c_str());

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code
        if (!prb->inplace) {
            placeholder_d_src_dt = make_dnn_mem(outs[0], (prb->tag[0]).c_str());
        }
        dnn_mem_t &d_src_dt = prb->inplace ? d_dst_dt : placeholder_d_src_dt;

        if (prepare_bwd(prb, src_fp, d_dst_fp, mean_fp, var_fp, sc_fp) != OK) {
            cleanup();
            return res->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        SAFE(d_dst_dt.reorder(d_dst_fp), WARN);
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
        if (prb->use_sc() && prb->use_sh()) {
            SAFE(sc_dt.reorder(sc_fp), WARN);
            SAFE(sh_dt.reorder(sh_fp), WARN);
        }

        const dnnl::graph::engine &eng = get_test_engine();

        dnnl::graph::tensor src_tensor(
                ins[0], eng, static_cast<void *>(src_dt));
        dnnl::graph::tensor d_dst_tensor(
                ins[1], eng, static_cast<void *>(d_dst_dt));
        dnnl::graph::tensor mean_tensor(
                ins[2], eng, static_cast<void *>(mean_dt));
        dnnl::graph::tensor var_tensor(
                ins[3], eng, static_cast<void *>(var_dt));

        dnnl::graph::tensor d_src_tensor(
                outs[0], eng, static_cast<void *>(d_src_dt));

        dnnl::graph::tensor gamma_tensor, beta_tensor, d_gamma_tensor,
                d_beta_tensor;

        auto gamma_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto beta_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto d_gamma_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        auto d_beta_v = make_dnn_mem(lt_sh, dt::f32, tag::x);
        if (prb->use_sc() && prb->use_sh()) {
            for (int64_t i = 0; i < prb->c; i++) {
                gamma_v.set_elem(i, sc_dt.get_elem(i));
                beta_v.set_elem(i, sh_dt.get_elem(i));
            }
            gamma_tensor = dnnl::graph::tensor(
                    ins[4], eng, static_cast<void *>(gamma_v));
            beta_tensor = dnnl::graph::tensor(
                    ins[5], eng, static_cast<void *>(beta_v));
            d_gamma_tensor = dnnl::graph::tensor(
                    outs[1], eng, static_cast<void *>(d_gamma_v));
            d_beta_tensor = dnnl::graph::tensor(
                    outs[2], eng, static_cast<void *>(d_beta_v));
        }

        tensors_in.push_back(src_tensor);
        tensors_in.push_back(d_dst_tensor);
        tensors_in.push_back(mean_tensor);
        tensors_in.push_back(var_tensor);

        tensors_out.push_back(d_src_tensor);

        if (prb->use_sc() && prb->use_sh()) {
            tensors_in.push_back(gamma_tensor);
            tensors_in.push_back(beta_tensor);

            tensors_out.push_back(d_gamma_tensor);
            tensors_out.push_back(d_beta_tensor);
        }

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (prb->use_sc() && prb->use_sh()) {
            for (int64_t i = 0; i < prb->c; i++) {
                d_sc_dt.set_elem(i, d_gamma_v.get_elem(i));
                d_sh_dt.set_elem(i, d_beta_v.get_elem(i));
            }
        }

        if (is_bench_mode(CORR)) {
            args_t args, ref_args;

            args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_MEAN, mean_fp);
            ref_args.set(DNNL_ARG_VARIANCE, var_fp);
            ref_args.set(DNNL_ARG_SCALE, sc_fp);
            ref_args.set(DNNL_ARG_SHIFT, sh_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);
            ref_args.set(DNNL_ARG_DIFF_SCALE, d_sc_fp);
            ref_args.set(DNNL_ARG_DIFF_SHIFT, d_sh_fp);
            ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_fp);
            ref_args.set(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_fp);

            std::vector<data_kind_t> kinds {SRC};
            if (prb->use_sc() && prb->use_sh() && (prb->dir & FLAG_WEI)) {
                args.set(DNNL_ARG_DIFF_SCALE, d_sc_dt);
                args.set(DNNL_ARG_DIFF_SCALE, d_sh_dt);
                kinds.push_back(SC);
                kinds.push_back(SH);
            }

            check_correctness(
                    prb, kinds, args, ref_args, ::lnorm::setup_cmp, res);
        }
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace lnorm
} // namespace benchdnnext