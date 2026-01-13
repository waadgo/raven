import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import pytorch_lightning as pl

def get_fsdp_strategy():
    """Returns a properly configured FSDP strategy for PyTorch Lightning"""
    # Mixed precision settings
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy (wrap modules with >1M parameters)
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1000000)
    
    # Create FSDP settings dict
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mp_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "forward_prefetch": True,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "activation_checkpointing": True,
        # State dict settings for checkpointing
        "state_dict_type": StateDictType.FULL_STATE_DICT,
        "state_dict_config": FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    }
    
    return fsdp_kwargs

def convert_batchnorm_and_layernorm_to_fp32(module):
    """Convert BatchNorm and LayerNorm to FP32 for stable training"""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                          nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                          nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, 
                          nn.InstanceNorm2d, nn.InstanceNorm3d)):
        module.float()  # Convert to float32
    
    for child in module.children():
        convert_batchnorm_and_layernorm_to_fp32(child)
    
    return module
