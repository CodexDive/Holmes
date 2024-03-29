# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np
import json
import torch

from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from .global_vars import get_args, get_tokenizer
from .utils import unwrap_model, print_rank_0
from megatron.fs_utils import (
    ensure_local_directory_exists,
    create_read_file_system,
    create_write_file_system,
)

_CHECKPOINT_VERSION = None
CHECKPOINT_META_FILENAME = "checkpoint_meta.json"
DISTRIBUTED_OPTIMIZER_CHECKPOINT_NAME = "distrib_optim.pt"
MODEL_CHECKPOINT_NAME = "model_optim_rng.pt"
TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None):
        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name)
        args_value = getattr(args, arg_name)
        error_message = (
            "{} value from checkpoint ({}) is not equal to the "
            "input argument value ({}).".format(arg_name, checkpoint_value, args_value)
        )
        assert checkpoint_value == args_value, error_message

    _compare("num_layers")
    _compare("hidden_size")
    _compare("num_attention_heads")
    _compare("add_position_embedding")
    if args.vocab_file:
        _compare("max_position_embeddings")
        _compare("make_vocab_size_divisible_by")
        _compare("padded_vocab_size")
        _compare("tokenizer_type")
    if args.data_parallel_random_init:
        _compare("data_parallel_random_init")
    if get_checkpoint_version() < 3.0:
        _compare("tensor_model_parallel_size", old_arg_name="model_parallel_size")
    if get_checkpoint_version() >= 3.0:
        _compare("tensor_model_parallel_size")
        _compare("pipeline_model_parallel_size")


def get_latest_checkpoint_path(
        checkpoints_path,
        iteration,
        release=False,
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(
            checkpoints_path, directory, f"mp_rank_{tensor_rank:02d}"
        )
    else:
        common_path = os.path.join(
            checkpoints_path,
            directory,
            f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}",
        )

    return common_path


def get_rank_directory_name(
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
        data_parallel_rank=None,
):
    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    dp_suffix = "" if data_parallel_rank is None else f"_dp_{data_parallel_rank:03d}"
    if not pipeline_parallel:
        return f"mp_rank_{tensor_rank:02d}" + dp_suffix
    return f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}" + dp_suffix


def get_inner_directory_name_with_fallback(
    file_system,
    iteration,
    release=False,
    pipeline_parallel=None,
    tensor_rank=None,
    pipeline_rank=None,
    data_parallel_rank=None,
    enable_fallback=False,
):
    """Return relative path to load/save model checkpoint file based on data_parallel_rank config.
    During loading, if not found, it's possible that the checkpoint is generated with a different config.
    Fallback to check other possible paths. If still not found, return empty string.
    """
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)

    res = os.path.join(
        directory,
        get_rank_directory_name(
            pipeline_parallel, tensor_rank, pipeline_rank, data_parallel_rank
        ),
        "",
    )
    if not enable_fallback:
        return res

    if file_system.exists(res):
        return res
    new_data_parallel_rank = None
    if data_parallel_rank is None:
        new_data_parallel_rank = 0
    res = os.path.join(
        directory,
        get_rank_directory_name(
            pipeline_parallel, tensor_rank, pipeline_rank, new_data_parallel_rank
        ),
        "",
    )
    if not file_system.exists(res):
        print(
            f"Checkpoint not found for file {res} at root {file_system.get_root_dir()}"
        )
        return ""
    print(f"Returning fallback checkpoint path {res}")
    return res


def get_checkpoint_relative_name_with_fallback(
    file_system,
    iteration,
    release=False,
    pipeline_parallel=None,
    tensor_rank=None,
    pipeline_rank=None,
    data_parallel_rank=None,
    enable_fallback=False,
) -> str:
    """Return relative path to load model checkpoint file based on data_parallel_rank config.
    If not found, it's possible that the checkpoint is generated with a different config.
    Fallback to check other possible paths.
    If still not found, return empty string.
    """
    # Try primary path
    return os.path.join(
        get_inner_directory_name_with_fallback(
            file_system,
            iteration,
            release,
            pipeline_parallel,
            tensor_rank,
            pipeline_rank,
            data_parallel_rank,
            enable_fallback,
        ),
        MODEL_CHECKPOINT_NAME,
    )

def find_checkpoint_rank_0(
    file_system, iteration, release=False, distributed_checkpointing=False
):
    """Finds the checkpoint for rank 0 without knowing if we are using
    pipeline parallelism or not.

    Since the checkpoint naming scheme changes if pipeline parallelism
    is present, we need to look for both naming schemes if we don't
    know if the checkpoint has pipeline parallelism.
    """

    # Look for checkpoint with no pipelining
    filename = get_checkpoint_relative_name_with_fallback(
        file_system,
        iteration,
        release,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
        data_parallel_rank=0 if distributed_checkpointing else None,
        enable_fallback=True,
    )
    if file_system.exists(filename):
        return filename

    # Look for checkpoint with pipelining
    filename = get_checkpoint_relative_name_with_fallback(
        file_system,
        iteration,
        release,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
        data_parallel_rank=0 if distributed_checkpointing else None,
        enable_fallback=True,
    )
    if file_system.exists(filename):
        return filename

    return None


def read_metadata(tracker_file):
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    metastring = tracker_file.read().strip()
    try:
        iteration = int(metastring)
    except ValueError:
        release = metastring == "release"
        if not release:
            print_rank_0(
                "ERROR: Invalid metadata file. Exiting"
            )
            sys.exit()
    assert iteration > 0 or release, "error parsing metadata file"
    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.cuda.LongTensor([iteration])
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        # We should now have all the same iteration.
        # If not, print a warning and chose the maximum
        # iteration across all ranks.
        if iteration != max_iter:
            print(
                "WARNING: on rank {} found iteration {} in the "
                "metadata while max iteration across the ranks "
                "is {}, replacing it with max iteration.".format(
                    torch.distributed.get_rank(), iteration, max_iter
                ),
                flush=True,
            )
    else:
        # When loading a checkpoint outside of training (for example,
        # when editing it), we might not have torch distributed
        # initialized, in this case, just assume we have the latest
        max_iter = iteration
    return max_iter, release


def get_rng_state():
    """collect rng state across data parallel ranks"""
    args = get_args()
    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }

    rng_state_list = None
    if (
            torch.distributed.is_initialized()
            and mpu.get_data_parallel_world_size() > 1
            and args.data_parallel_random_init
    ):
        rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list, rng_state, group=mpu.get_data_parallel_group()
        )
    else:
        rng_state_list = [rng_state]

    return rng_state_list


# TODO: Remove this method with fallback once compatibility is no longer needed.
def validate_existence_and_get_tracker_file(file_system, rank0=False):
    if file_system.exists(TRACKER_FILENAME, is_path_under_root=True):
        return file_system.get_filelike_reader(TRACKER_FILENAME, "r", is_path_under_root=True)
    print(
        f"Cannot find tracker file in expected location {file_system.get_root_dir()}, "
        f"falling back to {file_system.get_checkpoint_dir()}."
    )
    if not file_system.exists(TRACKER_FILENAME, is_path_under_root=False):
        print(
            f"Cannot find tracker file in fallback location {file_system.get_root_dir()}"
        )
        if not rank0:
            print(f"WARNING: could not find the metadata file {TRACKER_FILENAME}")
            print("    will not load any checkpoints and will start from " "random")
        return None
    return file_system.get_filelike_reader(TRACKER_FILENAME, "r", is_path_under_root=False)


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, train_ds=None):
    """Save a model checkpoint."""
    args = get_args()
    print_rank_0(
        "saving checkpoint at iteration {:7d} to {}".format(iteration, args.save)
    )

    file_system = create_write_file_system(args, relative_dir="")
    print(f"distributed_checkpointing: {args.distributed_checkpointing}")
    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer or args.overlapped_distributed_optimizer:
        optim_checkpoint_name = os.path.join(
            get_inner_directory_name_with_fallback(
                file_system,
                iteration,
                data_parallel_rank=mpu.get_data_parallel_rank()
                if args.distributed_checkpointing
                else None,
                enable_fallback=False,
            ),
            DISTRIBUTED_OPTIMIZER_CHECKPOINT_NAME,
        )
        print(f"saving distributed optimizer states to {optim_checkpoint_name}")
        if args.distributed_checkpointing:
            optimizer.save_parameter_state_in_parallel(
                file_system.get_filelike_writer(optim_checkpoint_name, mode="wb")
            )
        else:
            optimizer.save_parameter_state(
                file_system.get_filelike_writer(optim_checkpoint_name, mode="wb")
                if mpu.get_data_parallel_rank() == 0
                else None
            )

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state() if not args.no_save_rng else None
    # Save args, model params, non-sharded optimizer states, optimizer scheduler, rng states
    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0:
        state_dict = {"args": args, "checkpoint_version": 3.0, "iteration": iteration}
        model = unwrap_model(model)
        if len(model) == 1:
            state_dict["model"] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict["model%d" % i] = model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict["optimizer"] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                state_dict["opt_param_scheduler"] = opt_param_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state

        model_checkpoint_name = os.path.join(
            get_inner_directory_name_with_fallback(
                file_system,
                iteration,
                data_parallel_rank=mpu.get_data_parallel_rank()
                if args.distributed_checkpointing
                else None,
                enable_fallback=False,
            ),
            MODEL_CHECKPOINT_NAME,
        )
        torch.save(state_dict, file_system.get_filelike_writer(model_checkpoint_name, mode="wb"))

    # Saving tokenizer
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tokenizer = get_tokenizer()
        if hasattr(tokenizer, "tokenizer"):
            ensure_local_directory_exists(args.save)
            tokenizer.tokenizer.save_pretrained(args.save)
        else:
            print("no tokenizer to save")

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        "  successfully saved checkpoint at iteration {:7d} to {}".format(
            iteration, args.save
        )
    )

    # And update the latest iteration
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with file_system.get_filelike_writer(
                TRACKER_FILENAME, "w", is_path_under_root=True
        ) as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, "module"):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = (
        attention_module.num_attention_heads_per_partition
    )
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h]"""

        intermediate_shape = (
                                 num_splits,
                                 num_attention_heads_per_partition,
                                 hidden_size_per_attention_head,
                             ) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h]"""

        intermediate_shape = (
                                 num_attention_heads_per_partition,
                                 hidden_size_per_attention_head,
                                 num_splits,
                             ) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model) == 1
            model = model[0]
        for name, param in model.named_parameters():
            if name.endswith((".query_key_value.weight", ".query_key_value.bias")):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith((".key_value.weight", ".key_value.bias")):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(
            " succesfully fixed query-key-values ordering for"
            " checkpoint version {}".format(checkpoint_version)
        )


def _load_base_checkpoint(file_system, rank0=False, distributed_checkpointing=False):
    """Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_file = validate_existence_and_get_tracker_file(file_system, rank0=rank0)
    # If no tracker file, return nothing
    if tracker_file is None:
        return None, False
    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_file)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(
            file_system,
            iteration,
            release,
            distributed_checkpointing=distributed_checkpointing,
        )
    else:
        checkpoint_name = get_checkpoint_relative_name_with_fallback(
            file_system,
            iteration,
            release,
            data_parallel_rank=0 if distributed_checkpointing else None,
            enable_fallback=True,
        )
    if not file_system.exists(checkpoint_name):
        print(
            f"WARNING: could not find the checkpoint file {checkpoint_name} at root {file_system.get_root_dir()}."
            f"release:{release}, iteration:{iteration}"
        )
        return None, False
    print_rank_0(
        f"loading checkpoint from root {file_system.get_root_dir()}, release: {release}, iteration: {iteration}"
    )
    checkpoint_file = file_system.get_filelike_reader(checkpoint_name, "rb")
    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_file, map_location="cpu")
    except ModuleNotFoundError:
        from megatron.fp16_deprecated import loss_scaler

        # For backward compatibility.
        if not rank0:
            print_rank_0(" > deserializing using the old code structure ...")
        sys.modules["fp16.loss_scaler"] = sys.modules[
            "megatron.fp16_deprecated.loss_scaler"
        ]
        sys.modules["megatron.fp16.loss_scaler"] = sys.modules[
            "megatron.fp16_deprecated.loss_scaler"
        ]
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        sys.modules.pop("fp16.loss_scaler", None)
        sys.modules.pop("megatron.fp16.loss_scaler", None)
    except BaseException as e:
        print_rank_0("could not load the checkpoint")
        print_rank_0(e)
        sys.exit()

    return state_dict, release


def load_args_from_checkpoint(args, load_arg="load"):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print_rank_0("No load directory specified, using provided arguments.")
        return args
    file_system = create_read_file_system(args, relative_dir=None)
    state_dict, release = _load_base_checkpoint(file_system, rank0=True)

    # Args.
    if not state_dict:
        print_rank_0(
            "Checkpoint not found to provide arguments, using provided arguments."
        )
        return args

    if "args" not in state_dict:
        print_rank_0(
            "Checkpoint provided does not have arguments saved, using provided arguments."
        )
        return args

    checkpoint_args = state_dict["args"]
    checkpoint_version = state_dict.get("checkpoint_version", 0)
    args.iteration = state_dict["iteration"]

    # One-off conversion for foundation models
    if hasattr(checkpoint_args, "disable_bias_linear"):
        setattr(
            checkpoint_args,
            "add_bias_linear",
            not getattr(checkpoint_args, "disable_bias_linear"),
        )

    def _set_arg(arg_name, old_arg_name=None, force=False):
        if not force and getattr(args, arg_name, None) is not None:
            return

        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name, None)

        if checkpoint_value is not None:
            print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
            setattr(args, arg_name, checkpoint_value)
        else:
            print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

    _set_arg("num_layers")
    _set_arg("hidden_size")
    _set_arg("ffn_hidden_size")
    _set_arg("seq_length")
    _set_arg("num_attention_heads")
    _set_arg("kv_channels")
    _set_arg("max_position_embeddings")
    _set_arg("add_position_embedding", force=True)
    _set_arg("use_rotary_position_embeddings", force=True)
    _set_arg("rotary_percent", force=True)
    _set_arg("add_bias_linear", force=True)
    _set_arg("swiglu", force=True)
    _set_arg("untie_embeddings_and_output_weights", force=True)
    _set_arg("apply_layernorm_1p", force=True)
    _set_arg("tokenizer_type")
    _set_arg("padded_vocab_size")
    if checkpoint_version < 3.0:
        _set_arg("tensor_model_parallel_size", "model_parallel_size")
    else:
        _set_arg("tensor_model_parallel_size", force=True)
        _set_arg("pipeline_model_parallel_size", force=True)
        _set_arg("virtual_pipeline_model_parallel_size", force=True)
        _set_arg("num_layers_per_virtual_pipeline_stage")
    return args, checkpoint_args


def load_checkpoint(
        model, optimizer, opt_param_scheduler, strict=True, load_dir=None
):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    file_system = create_read_file_system(args, relative_dir=load_dir)
    state_dict, release = _load_base_checkpoint(
        file_system,
        rank0=False,
        distributed_checkpointing=args.distributed_checkpointing,
    )

    # Checkpoint not loaded.
    if state_dict is None:
        # Conditionally exit at this point.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            torch.distributed.barrier()
            sys.exit()

        # Iteration defaults to 0.
        return 0

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict["iteration"]
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict["total_iters"]
            except KeyError:
                print_rank_0(
                    "A metadata file exists but unable to load "
                    "iteration from checkpoint, exiting"
                )
                sys.exit()

    _load_common_from_state_dict(args, release, state_dict, model, strict=strict)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if optimizer is not None:
                optimizer.load_state_dict(state_dict["optimizer"])
            # Load distributed optimizer's custom parameter state.
            if args.use_distributed_optimizer or args.overlapped_distributed_optimizer:
                tracker_file = validate_existence_and_get_tracker_file(file_system, rank0=False)
                assert tracker_file is not None, "no tracker file found"
                tmp_iteration, tmp_release = read_metadata(tracker_file)
                optim_checkpoint_name = os.path.join(
                    get_inner_directory_name_with_fallback(
                        file_system,
                        tmp_iteration,
                        tmp_release,
                        data_parallel_rank=mpu.get_data_parallel_rank()
                        if args.distributed_checkpointing
                        else None,
                        enable_fallback=True,
                    ),
                    DISTRIBUTED_OPTIMIZER_CHECKPOINT_NAME,
                )
                if args.distributed_checkpointing:
                    optimizer.load_parameter_state_in_parallel(
                        file_system.get_filelike_reader(optim_checkpoint_name, mode="rb")
                    )
                else:
                    optimizer.load_parameter_state(
                        file_system.get_filelike_reader(optim_checkpoint_name, mode="rb")
                    )

            # Load scheduler.
            if opt_param_scheduler is not None:
                if "lr_scheduler" in state_dict:  # backward compatibility
                    opt_param_scheduler.load_state_dict(state_dict["lr_scheduler"])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-optim or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(file_system.get_root_dir())
            )
            sys.exit()
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {args.load} at iteration {iteration}"
    )

    return iteration


def load_biencoder_checkpoint(
        model, only_query_model=False, only_context_model=False, custom_load_path=None
):
    """
    selectively load retrieval models for indexing/retrieving
    from saved checkpoints
    """

    args = get_args()
    model = unwrap_model(model)

    load_path = custom_load_path if custom_load_path is not None else args.load
    file_system = create_read_file_system(args, relative_dir=None, custom_root_dir=load_path)
    tracker_file = validate_existence_and_get_tracker_file(file_system, rank0=False)
    assert tracker_file is not None, "no tracker file found"
    iteration, _ = read_metadata(tracker_file)

    checkpoint_name = get_checkpoint_relative_name_with_fallback(
        file_system,
        iteration,
        release=False,
        enable_fallback=False,
    )
    if mpu.get_data_parallel_rank() == 0:
        print(
            "global rank {} is loading checkpoint {}".format(
                torch.distributed.get_rank(), checkpoint_name
            )
        )

    checkpoint_file = file_system.get_filelike_reader(checkpoint_name, "rb")
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    ret_state_dict = state_dict["model"]

    if only_query_model:
        ret_state_dict.pop("context_model")
    if only_context_model:
        ret_state_dict.pop("query_model")

    assert len(model) == 1
    model[0].load_state_dict(ret_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(" successfully loaded {}".format(checkpoint_name))

    return model


def _load_common_from_state_dict(args, release, state_dict, model, strict=True):
    # Set checkpoint version.
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if "args" in state_dict and not args.finetune:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(
            checkpoint_args, "consumed_train_samples", 0
        )
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(
            checkpoint_args, "consumed_valid_samples", 0
        )
    else:
        print_rank_0("load_common_from_state_dict: Could not find arguments in the checkpoint ...")

    # Model.
    model = unwrap_model(model)
    assert len(model) == 1, "PP Checkpointing not supported yet."
    model[0].load_state_dict(state_dict["model"], strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")
    fix_query_key_value_ordering(model, checkpoint_version)

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if "rng_state" in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict["rng_state"][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict["rng_state"][0]
                random.setstate(rng_state["random_rng_state"])
                np.random.set_state(rng_state["np_rng_state"])
                torch.set_rng_state(rng_state["torch_rng_state"])
                torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
                # Check for empty states array
                if not rng_state["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state["rng_tracker_states"]
                )
            else:  # backward compatibility
                random.setstate(state_dict["random_rng_state"])
                np.random.set_state(state_dict["np_rng_state"])
                torch.set_rng_state(state_dict["torch_rng_state"])
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
                # Check for empty states array
                if not state_dict["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict["rng_tracker_states"]
                )
            print("Loaded rng states")
        except KeyError:
            print_rank_0(
                "Unable to load rng state from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the rng state, "
                "exiting ..."
            )
            sys.exit()
