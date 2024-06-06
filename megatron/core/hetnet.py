import os
import torch

from tools.config import nccl_config
import datetime



def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_rank_last(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            print(message, flush=True)
    else:
        print(message, flush=True)
        

def set_nccl_envs(device_name):
    if os.getenv("NCCL_SOCKET_IFNAME") is None:
        raise RuntimeError("NCCL_SOCKET_IFNAME was not set")
    
    os.unsetenv("NCCL_IB_DISABLE")
    os.unsetenv("NCCL_IBEXT_DISABLE")
    os.unsetenv("NCCL_SOCKET_IFNAME")
    os.unsetenv("NCCL_IB_HCA")
    os.unsetenv("NCCL_NET_GDR_LEVEL")
    os.unsetenv("NCCL_NET")
    os.unsetenv("NCCL_COMM_ID")
    
    if nccl_config[device_name].get("NCCL_NET"):
        os.environ["NCCL_NET"]=nccl_config[device_name].get("NCCL_NET")
    if nccl_config[device_name].get("NCCL_IB_DISABLE"):
        os.environ["NCCL_IB_DISABLE"]=nccl_config[device_name].get("NCCL_IB_DISABLE")
    if nccl_config[device_name].get("NCCL_IBEXT_DISABLE"):
        os.environ["NCCL_IBEXT_DISABLE"]=nccl_config[device_name].get("NCCL_IBEXT_DISABLE")
    if nccl_config[device_name].get("NCCL_NET_GDR_LEVEL"):
        os.environ["NCCL_NET_GDR_LEVEL"]=nccl_config[device_name].get("NCCL_NET_GDR_LEVEL")
    if nccl_config[device_name].get("NCCL_SOCKET_IFNAME"):
        os.environ["NCCL_SOCKET_IFNAME"]=nccl_config[device_name].get("NCCL_SOCKET_IFNAME")
    if nccl_config[device_name].get("NCCL_IB_HCA"):
        os.environ["NCCL_IB_HCA"]=nccl_config[device_name].get("NCCL_IB_HCA")
    

def init_nccl_net(group):
    temp = torch.ones(1, device="cuda")
    torch.distributed.all_reduce(temp, group = group)
    torch.cuda.synchronize()


def new_nccl_group(ranks,backend="nccl",timeout=None,pg_options=None,device_name=None):

    print_rank_0("use_{0}_env".format(device_name))
    print_rank_0(list(ranks))
    print_rank_last("use_{0}_env".format(device_name))
    print_rank_last(list(ranks))

    set_nccl_envs(device_name)
    group = torch.distributed.new_group(ranks, backend=backend,timeout=timeout, pg_options=pg_options)
    init_nccl_net(group=group)
    return group


def new_process_group(ranks, backend = "nccl",timeout=None,pg_options=None,device_types=None):
    if backend =="nccl":
        if len(device_types)== 1:
            device_name=next(iter(device_types))
        else:
            device_name="heter"

        return new_nccl_group(ranks,backend="nccl",timeout=timeout,pg_options=pg_options, device_name=device_name)
    
    else:
        return torch.distributed.new_group(ranks, backend=backend)
