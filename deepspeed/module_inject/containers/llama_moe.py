# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features import HybridSplitQKVContainer, HybridGatedMLPContainer, MetaTensorContainer
from deepspeed.utils.types import ActivationFuncType, NormType
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter

from ..policy import (
    TransformerPolicy,
    transformer_param_names,
    maybe_copy,
    maybe_copy_qkv,
    maybe_copy_geglu,
    maybe_get_lora,
    maybe_copy_experts_geglu,
    maybe_copy_experts,
)


class DS_MixtralContainer(MetaTensorContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        _config.rotate_half = True
        _config.rotate_every_two = False
        _config.rotary_dim = self.hidden_size // self.num_attention_heads
        _config.rope_theta = self.policy.client_module.self_attn.rope_theta
        _config.n_experts = self.policy.client_module.self_attn.config.num_local_experts
        _config.n_top_k = self.policy.client_module.self_attn.config.num_experts_per_tok
        _config.moe_freq = self.policy.client_module.self_attn.config.moe_layer_frequency
        # _config.num_kv = 8
        # _config.multi_query = True
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)

        return self.module

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        return
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.block_sparse_moe.w2.weight, self.policy.client_module.block_sparse_moe.w1.weight,
                self.policy.client_module.block_sparse_moe.w3.weight, self.policy.client_module.self_attn.q_proj.weight,
                self.policy.client_module.self_attn.k_proj.weight, self.policy.client_module.self_attn.v_proj.weight,
                self.policy.client_module.self_attn.o_proj.weight
            ]
        ]

    def get_lora_matched_pair(self):
        return
        up_proj_lora, gate_proj_lora, down_proj_lora, q_lora, k_lora, v_lora, out_lora = self.get_lora_params()
        ret = [(up_proj_lora, self.inter_up_w), (gate_proj_lora, self.inter_gate_w), (down_proj_lora, self._4hh_w),
               (out_lora, self.dense_w), (q_lora, self.qw), (k_lora, self.kw), (v_lora, self.vw)]
        return ret

    def set_q_k_v(self):
        """
        Necessary to implement for `HybridSplitQKVContainer`
        """
        return
        self.qw = self.policy.client_module.self_attn.q_proj.weight
        self.qb = None
        self.kw = self.policy.client_module.self_attn.k_proj.weight
        self.kb = None
        self.vw = self.policy.client_module.self_attn.v_proj.weight
        self.vb = None

    def set_mlp_gate(self):
        """
        Necessary to implement for `HybridGatedMLPContainer`
        """
        return
        self.inter_up_w = self.policy.client_module.block_sparse_moe.w3.weight
        self.inter_up_b = None
        self.inter_gate_w = self.policy.client_module.block_sparse_moe.w1.weight
        self.inter_gate_b = None

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        if module.moe_layer:
            param_names = (
                'self_attn.q_proj.weight', \
                'self_attn.k_proj.weight', \
                'self_attn.v_proj.weight', \
                'self_attn.o_proj.weight', \
            ) + \
            tuple(
                    [
                        (
                            f'block_sparse_moe.experts.{i}.w3.weight',
                            f'block_sparse_moe.experts.{i}.w1.weight',
                        ) for i in range(module.n_experts)
                    ]
            ) + \
            tuple(
                    [f'block_sparse_moe.experts.{i}.w2.weight' for i in range(module.n_experts)]
            ) + \
            (
                'block_sparse_moe.gate.weight', \
                'post_attention_layernorm.weight', \
                'input_layernorm.weight',
            ) 
            for expert_id in range(module.n_experts):
                maybe_copy_experts_geglu(module.mlp, sd, weight_quantizer, mp_replace, 'inter_w', prefix, param_names[4 + expert_id], expert_id)
                maybe_copy_experts(module.mlp, sd, weight_quantizer, mp_replace, 'output_w', prefix + param_names[4 + module.n_experts + expert_id], expert_id=expert_id)
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, 'gate_w', prefix + param_names[-3])
        else:
            param_names = (
                'self_attn.q_proj.weight', \
                'self_attn.k_proj.weight', \
                'self_attn.v_proj.weight', \
                'self_attn.o_proj.weight', \
                'block_sparse_moe.w3.weight', \
                'block_sparse_moe.w1.weight', \
                'block_sparse_moe.w2.weight', \
                'post_attention_layernorm.weight', \
                'input_layernorm.weight',
            )
            maybe_copy_geglu(module.mlp, sd, weight_quantizer, mp_replace, 'inter_w',
                            [prefix + param_names[4], prefix + param_names[5]])
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, 'output_w', prefix + param_names[6])


        maybe_copy_qkv(module.attention,
                    sd,
                    weight_quantizer,
                    mp_replace,
                    'attn_qkvw', [prefix + param_names[0], prefix + param_names[1], prefix + param_names[2]],
                    split_qkv=self.policy.split_qkv)
        for i in range(3, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i - 1],
                    prefix + param_names[i])
        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[8], prefix + param_names[-2])
        maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[10], prefix + param_names[-1])

        # This line is necessary for proper output when kernels + meta tensors are used in Llama models
        # TODO: Investigate root-cause and fix meta tensor loading
        module.mlp.output_b = None
        module.mlp.inter_b = None
        module.attention.attn_qkvb = None
        module.attention.attn_ob = None


class MixtralLayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=True):
        super().__init__(
            inference,
            mlp_act_func_type=ActivationFuncType.GATED_SILU,
            norm_type=NormType.RMSNorm,
        )
        self.client_module = client_module
        try:
            import transformers
            MixtralLayerPolicy._orig_layer_class = [
                transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer,
            ]  # type: ignore
        except:
            MixtralLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if self.client_module.moe_layer:
            experts = self.client_module.block_sparse_moe.experts
            hidden_heads = (
                getattr(self.client_module.self_attn.q_proj.weight, "ds_shape",
                        self.client_module.self_attn.q_proj.weight.shape)[1],
                self.client_module.self_attn.num_heads,
                self.client_module.input_layernorm.variance_epsilon,
                experts[0].w1.weight.shape[0],
            )
        else:
            hidden_heads = (
                getattr(self.client_module.self_attn.q_proj.weight, "ds_shape",
                        self.client_module.self_attn.q_proj.weight.shape)[1],
                self.client_module.self_attn.num_heads,
                self.client_module.input_layernorm.variance_epsilon,
                getattr(self.client_module.block_sparse_moe.w1.weight, "ds_shape", self.client_module.block_sparse_moe.w1.weight.shape)[0],
            )
        return hidden_heads

    def attention(self, enable_training=False):
        qw = self.client_module.self_attn.q_proj.weight
        kw = self.client_module.self_attn.k_proj.weight
        vw = self.client_module.self_attn.v_proj.weight

        # num_kv = self.client_module.self_attn.num_key_value_heads
        # head_dim = self.client_module.self_attn.head_dim
        # qw = qw.reshape(num_kv, -1, head_dim, qw.shape[-1])
        # kw = kw.reshape(num_kv, -1, head_dim, kw.shape[-1])
        # vw = vw.reshape(num_kv, -1, head_dim, kw.shape[-1])

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)
        # qkvw = Parameter(torch.cat((qw, kw, vw), dim=1), requires_grad=enable_training).reshape(-1, qw.shape[-1])

        return qkvw, \
                None, \
                self.client_module.self_attn.o_proj.weight, \
                None

    def mlp(self, enable_training=False):
        if self.client_module.moe_layer:
            mlp1 = []
            mlp2 = []
            for expert in self.client_module.block_sparse_moe.experts:
                mlp1_up = expert.w3.weight
                mlp1_gate = expert.w1.weight
            
                mlp1.append(torch.cat((mlp1_up, mlp1_gate), dim=0))
                mlp2.append(expert.w2.weight)
            mlp1 = torch.stack(mlp1, dim=0)
            mlp2 = torch.stack(mlp2, dim=0)
        else:
            mlp1_up = self.client_module.block_sparse_moe.w3.weight
            mlp1_gate = self.client_module.block_sparse_moe.w1.weight
            mlp2 = self.client_module.block_sparse_moe.w2.weight

            mlp1 = Parameter(torch.cat((mlp1_up, mlp1_gate), dim=0), requires_grad=enable_training)

        return mlp1, None, mlp2, None

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               None, \
               self.client_module.input_layernorm.weight, \
               None
