# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.utils.types import GATED_ACTIVATION_TYPES
from deepspeed.accelerator import get_accelerator
from .op_binding import MLPGemmOp, VectorMatMulOp, GELUGemmOp, ResidualAddOp
from deepspeed.inference.v2.kernels.ragged_ops import RaggedTopKGating, MoEScatter, MoEGather
from deepspeed.inference.v2.kernels.cutlass_ops import MoEGEMM
from .op_binding.base import BaseOp

def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[1]) * x[0]

class DeepSpeedMoEMLP(BaseOp):
    _inter_w_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False):
        super(DeepSpeedMoEMLP, self).__init__(config)

        self.n_experts = config.n_experts #// self.config.mp_size
        data_type = torch.int8 if self.config.dtype == torch.int8 else self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        device = get_accelerator().current_device_name()

        proj_factor = 2 if self.config.mlp_act_func_type in GATED_ACTIVATION_TYPES else 1
        self.config.intermediate_size = self.config.intermediate_size if self.config.intermediate_size > 0 else 4 * self.config.hidden_size
        self.intm_w_sz_per_partition = self.config.intermediate_size * proj_factor // self.config.mp_size
        self.intm_o_sz_per_partition = self.config.intermediate_size // self.config.mp_size

        if self.config.set_empty_params:
            self.attn_nw = None
            self.attn_nb = None
            self.inter_w = None
            self.inter_b = None
            self.inter_up_w = None
            self.inter_up_b = None
            self.inter_gate_w = None
            self.inter_gate_b = None
            self.output_w = None
            self.output_b = None
        else:
            self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)
            self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)

            self.inter_w = nn.Parameter(torch.empty(self.n_experts,
                                                    self.config.hidden_size,
                                                    self.intm_w_sz_per_partition,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)
            self.inter_b = nn.Parameter(torch.empty(self.n_experts, self.intm_w_sz_per_partition, dtype=data_type_fp, device=device),
                                        requires_grad=False)
            self.output_w = nn.Parameter(torch.empty(self.n_experts,
                                                     self.intm_o_sz_per_partition,
                                                     self.config.hidden_size,
                                                     dtype=data_type,
                                                     device=device),
                                         requires_grad=False)
            self.output_b = nn.Parameter(torch.empty(self.n_experts, self.config.hidden_size, dtype=data_type_fp, device=device),
                                         requires_grad=False)
            # self.n_experts *= self.config.mp_size
            
            self.gate_w = nn.Parameter(torch.empty(self.config.hidden_size, self.n_experts,
                                                    dtype=data_type_fp, device=device),
                                         requires_grad=False)
                                         
        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        self.mp_group = mp_group

        self.mlp_gemm_func = MLPGemmOp(config)
        self.norm_res_func = self.inference_module.ds_rms_norm_residual
        self.gated_activation = self.inference_module.gated_activation
        self.vector_matmul_func = VectorMatMulOp(config)
        self.fused_gemm_gelu = GELUGemmOp(config)
        self.residual_add_func = ResidualAddOp(config)

        self.n_top_k = self.config.n_top_k
        self.expert_counts = torch.empty((self.n_experts, ),
                                          dtype=torch.int32,
                                          device=get_accelerator().current_device())
        self.expert_cumsum = torch.empty((self.n_experts, ),
                                          dtype=torch.int64,
                                          device=get_accelerator().current_device())

        # self.n_experts = self.n_experts // self.config.mp_size

        self._top_1_gate = RaggedTopKGating(data_type)

        self._moe_scatter = MoEScatter(data_type, self.config.hidden_size)
        self._moe_gather = MoEGather(data_type, self.config.hidden_size, self.n_top_k > 1)

        self._moe_mlp = MoEGEMM(fp_dtype=data_type)

        if len(DeepSpeedMoEMLP._inter_w_buffers) == 0:
            DeepSpeedMoEMLP._inter_w_buffers = [
                torch.empty(self.intm_w_sz_per_partition, self.config.hidden_size, dtype=data_type, device=device),
                torch.empty(self.intm_w_sz_per_partition, dtype=data_type_fp, device=device)
            ]
        self.gen_tokens = 0
        
        
    def _merge_inter_w(self):
        inter_w = DeepSpeedMoEMLP._inter_w_buffers[0]
        inter_w[:self.intm_w_sz_per_partition // 2, :] = self.inter_up_w  # type: ignore
        inter_w[self.intm_w_sz_per_partition // 2:, :] = self.inter_gate_w  # type: ignore
        if self.inter_up_b is not None:
            inter_b = DeepSpeedMoEMLP._inter_w_buffers[1]
            inter_b[:self.intm_w_sz_per_partition // 2] = self.inter_up_b  # type: ignore
            inter_b[self.intm_w_sz_per_partition // 2:] = self.inter_gate_b  # type: ignore
        return DeepSpeedMoEMLP._inter_w_buffers

    def _run_moe(self, moe_input, scores, mapped_slots):
        tokens = moe_input.shape[0]
        gated_intermediate = torch.empty(
                (tokens, self.intm_w_sz_per_partition),
                dtype=moe_input.dtype,
                device=get_accelerator().current_device())
        output_unordered = torch.empty((tokens, self.config.hidden_size),
                                             dtype=moe_input.dtype,
                                             device=get_accelerator().current_device())
        output = torch.zeros((tokens // self.n_top_k, self.config.hidden_size),
                                   dtype=moe_input.dtype,
                                   device=get_accelerator().current_device())
        self._moe_mlp(
            gated_intermediate,
            moe_input, 
            self.inter_w,
            self.expert_cumsum, 
            self.inter_b,
        )
        intermediate = swiglu(gated_intermediate)
        # intermediate = self.gated_activation(gated_intermediate.unsqueeze(0), torch.tensor([]), 4).squeeze(0)
        self._moe_mlp(
            output_unordered, 
            intermediate, 
            self.output_w,
            self.expert_cumsum, 
            self.output_b,
        )
        self._moe_gather(output, output_unordered, scores, mapped_slots, self.expert_counts)
        return output

    def _moe_gating(self, input):
        hidden_states = input.reshape(-1, input.shape[-1])
        tokens = hidden_states.shape[0]
        logits = torch.matmul(hidden_states, self.gate_w.to(input.dtype))
        scores = torch.empty((tokens, self.n_top_k),
                                   dtype=torch.float32,
                                   device=get_accelerator().current_device())
        assignments = torch.empty((tokens, self.n_top_k),
                                        dtype=torch.int32,
                                        device=get_accelerator().current_device())
        offsets = torch.empty((tokens, self.n_top_k),
                                    dtype=torch.int32,
                                    device=get_accelerator().current_device())
        moe_input = torch.empty((tokens * self.n_top_k, self.config.hidden_size),
                                      dtype=input.dtype,
                                      device=get_accelerator().current_device())
        mapped_slots = torch.empty((tokens, self.n_top_k),
                                         dtype=torch.int32,
                                         device=get_accelerator().current_device())
        self.expert_counts.zero_()
        self._top_1_gate(self.expert_counts, scores, assignments, offsets, logits,)
        self._moe_scatter(
            moe_input, 
            self.expert_cumsum, 
            mapped_slots, 
            hidden_states, 
            self.expert_counts, 
            assignments, 
            offsets
        )
        return moe_input, scores, mapped_slots

    def forward(self, input, residual, residual_norm, bias):
        self.gen_tokens += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.inter_w is None:
            self._inter_w, self._inter_b = self._merge_inter_w()
        else:
            self._inter_w = self.inter_w
            self._inter_b = self.inter_b

        residual_add = None
        if self.attn_nw is None:
            output = self.fused_gemm_gelu(input=residual_norm,
                                          weight=self._inter_w,
                                          bias=self._inter_b,
                                          weight_out=self.output_w)
        else:
            norm_output, residual_add = self.norm_res_func(input,
                                                           residual,
                                                           self.attn_nw,
                                                           self.config.epsilon)
            if self.config.use_baseline_implementation:
                hidden_states = norm_output.reshape(-1, input.shape[-1])
                router_logits = torch.matmul(hidden_states, self.gate_w.to(input.dtype))

                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, self.n_top_k, dim=-1)

                # we cast back to the input dtype
                routing_weights = routing_weights.to(hidden_states.dtype)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

                final_hidden_states = torch.zeros(
                    hidden_states.shape, dtype=torch.float, device=hidden_states.device
                )
                expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_experts).permute(2, 1, 0)
                
                for expert_idx in range(self.n_experts):
                    intm_layer = self.inter_w[expert_idx]
                    out_layer = self.output_w[expert_idx]
                    idx, top_x = torch.where(expert_mask[expert_idx])

                    if top_x.shape[0] == 0:
                        continue

                    top_x_list = top_x.tolist()
                    idx_list = idx.tolist()
                    current_state = hidden_states[None, top_x_list].reshape(-1, input.shape[-1])
                    
                    current_hidden_states = torch.matmul(current_state, intm_layer) 
                    current_hidden_states = swiglu(current_hidden_states)
                    current_hidden_states = torch.matmul(current_hidden_states, out_layer)
                    current_hidden_states = current_hidden_states * routing_weights[top_x_list, idx_list, None]

                    final_hidden_states.index_add_(0, top_x, current_hidden_states.to(final_hidden_states.dtype))
                output = final_hidden_states.reshape(input.shape)
            else:
                moe_input, scores, mapped_slots = self._moe_gating(norm_output)
                output = self._run_moe(moe_input, scores, mapped_slots).reshape(input.shape)

        if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
           parallel_output = output.float() if self.config.fp32_allreduce else output
           dist.all_reduce(parallel_output, group=self.mp_group)
           output = parallel_output
        
        return (residual.float() + output).to(input.dtype)
        # residual = self.residual_add_func(hidden_state=output,
        #                                   residual=residual,
        #                                   add_bias=bias is not None,
        #                                   attention_output=input,
        #                                   attention_bias=bias if bias is not None else self.output_b,
        #                                   final_bias=self.output_b,
        #                                   residual_add=residual_add)
        # if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
        #     parallel_residual = residual.float() if self.config.fp32_allreduce else residual
        #     dist.all_reduce(parallel_residual, group=self.mp_group)
        #     residual = parallel_residual.to(residual.dtype)
        # return residual
