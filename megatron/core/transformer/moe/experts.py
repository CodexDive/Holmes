# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe import sparse_gemm_util as sg
from megatron.core.transformer.transformer_config import TransformerConfig


class SparseMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig, blocking=256):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        self.blocking = blocking
        self.gated_linear_unit = self.config.gated_linear_unit
        sg.assert_megablocks_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in block-sparse GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1

        self.activation_func = lambda x: sg.sparse_act(x, self.config.activation_func(x.data))

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.gated = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.gated,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.gated = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.gated,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.gated, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def sparse_transpose(self, size, row_indices, column_indices, offsets, dim_ff, blocking=256):
        block_columns = size[1] // blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input values.
        # To avoid overflow when we have large activation matrices we cast to
        # 32-bit before sorting.
        _, gather_indices = sg.ops.sort(
            column_indices.int(),
            max(int(np.ceil(np.log2((dim_ff * self.num_local_experts) // blocking))), 1))

        # There are a constant number of blocks in every row of the sparse matrix.
        # A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can divide
        # by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = sg.ops.histogram(column_indices, block_columns)
        nnz_per_column = sg.ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x, padded_bins, dim_ff, blocking=256):
        padded_tokens, _ = x.size()
        assert padded_tokens % blocking == 0
        assert dim_ff % blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // blocking
        blocks_per_row = dim_ff // blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device)

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = sg.ops.topology(padded_bins,
                                        blocking,
                                        block_rows,
                                        blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            blocking,
            blocking,
            dtype=x.dtype,
            device='meta')
        shape = (
            padded_tokens,
            dim_ff * self.num_local_experts
        )
        row_indices = sg.stk.ops.row_indices(
            shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, row_indices, column_indices, offsets, dim_ff, blocking)
        return sg.stk.Matrix(shape, data, row_indices, column_indices, offsets,
                            column_indices_t, offsets_t, block_offsets_t)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        if permuted_local_hidden_states.nelement() != 0:
            # padded hidden_states
            padded_blocks = []
            padded_tokens_per_expert = sg.ops.round_up(tokens_per_expert, self.blocking)
            padded_bins = sg.ops.inclusive_cumsum(padded_tokens_per_expert, 0)
            splited_permuted_local_hidden_states = torch.split(
                permuted_local_hidden_states, tokens_per_expert.tolist() , dim=0)
            for hidden_state, padding_size in zip(
                splited_permuted_local_hidden_states, padded_tokens_per_expert):
                pad_size = padding_size - hidden_state.size(0)
                padded_block = F.pad(hidden_state, (0, 0, 0, pad_size))
                padded_blocks.append(padded_block)
            padded_permuted_local_hidden_states = torch.cat(padded_blocks, dim=0)

            # calculate topo
            with torch.no_grad():
                dim_ff = self.weight1.size()[-1] // self.num_local_experts
                topo = self.topology(
                    padded_permuted_local_hidden_states, padded_bins,
                    dim_ff, self.blocking)
            if self.gated_linear_unit:
                padded_fc1_output = sg.stk.Matrix(
                    topo.size(),
                    sg.stk.ops.sdd(padded_permuted_local_hidden_states, self.weight1, topo).data * \
                        self.activation_func(sg.stk.ops.sdd(padded_permuted_local_hidden_states, self.gated, topo)).data,
                    topo.row_indices,
                    topo.column_indices,
                    topo.offsets,
                    topo.column_indices_t,
                    topo.offsets_t,
                    topo.block_offsets_t
                )
                padded_fc2_output = sg.stk.ops.dsd(padded_fc1_output, self.weight2)
            else:
                padded_fc1_output = sg.stk.ops.sdd(padded_permuted_local_hidden_states, self.weight1, topo)
                activation_x = self.activation_func(padded_fc1_output)
                padded_fc2_output = sg.stk.ops.dsd(activation_x, self.weight2)         
            
            # unpadded output
            splited_fc2_output = torch.split(padded_fc2_output, padded_tokens_per_expert.tolist(), dim=0)
            padded_fc2_output_append = []
            for index in range(tokens_per_expert.size()[0]):
                unpadded_splited_fc2_output = splited_fc2_output[index][:tokens_per_expert[index]]
                padded_fc2_output_append.append(unpadded_splited_fc2_output)
            fc2_output = torch.cat(padded_fc2_output_append, dim=0)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.config.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        raise NotImplementedError(
            'Currently distributed checkpointing is not supported for SparseMLP'
        )


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func != F.silu:
                raise ValueError("Activation function must be silu when using GroupedMLP.")

            @jit_fuser
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        raise NotImplementedError(
            'Currently distributed checkpointing is not supported for GroupedMLP'
        )


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.
    
    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output, output_bias = expert(hidden)

            output_local[start:end] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_local[start:end, :] = output_bias

        return output_local, output_bias_local

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """ Maps local expert to global experts. """
        sharded_state_dict = {}
        num_global_experts = (
            parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        expert_sharded_prefix = f'{prefix}experts.'
        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets, metadata
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(),
                )

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict
