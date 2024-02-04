import os
import torch
from megatron import get_args
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
        

def set_nccl_socket_envs():
    if os.getenv("NCCL_SOCKET_IFNAME") is None:
        raise RuntimeError("NCCL_SOCKET_IFNAME was not set")
    
    os.unsetenv("NCCL_IB_DISABLE")
    os.unsetenv("NCCL_IBEXT_DISABLE")
    os.unsetenv("NCCL_SOCKET_IFNAME")
    os.unsetenv("NCCL_IB_HCA")
    os.unsetenv("NCCL_NET_GDR_LEVEL")
    os.unsetenv("NCCL_NET")
    os.unsetenv("NCCL_COMM_ID")
    
    os.environ["NCCL_IB_DISABLE"]=nccl_config.DISABLE
    os.environ["NCCL_IBEXT_DISABLE"]=nccl_config.DISABLE
    os.environ["NCCL_SOCKET_IFNAME"]=nccl_config.SOCKET_IFNAME
    os.environ["NCCL_NET"]=nccl_config.NET_Socket
    

def set_nccl_ib_envs():
    args = get_args()
    
    os.unsetenv("NCCL_IB_DISABLE")
    os.unsetenv("NCCL_IBEXT_DISABLE")
    os.unsetenv("NCCL_IB_HCA")
    os.unsetenv("NCCL_SOCKET_IFNAME")
    os.unsetenv("NCCL_NET")
    os.unsetenv("NCCL_NET_GDR_LEVEL")
    os.unsetenv("NCCL_IB_GID_INDEX")
    
    if args.use_hetnet:
        os.environ["NCCL_NET"]=nccl_config.NET_IB
        os.environ["NCCL_IB_DISABLE"]=nccl_config.ENABLE
        os.environ["NCCL_IBEXT_DISABLE"]=nccl_config.ENABLE
        os.environ["NCCL_NET_GDR_LEVEL"]=nccl_config.NET_GDR_LEVEL
        os.environ["NCCL_SOCKET_IFNAME"]=nccl_config.SOCKET_IFNAME
        os.environ["NCCL_IB_HCA"]=nccl_config.IB_HCA
    else:
        os.environ["NCCL_IB_DISABLE"]=nccl_config.DISABLE
        os.environ["NCCL_IBEXT_DISABLE"]=nccl_config.DISABLE
        os.environ["NCCL_SOCKET_IFNAME"]=nccl_config.SOCKET_IFNAME
        os.environ["NCCL_NET"]=nccl_config.NET_Socket

  
def set_nccl_roce_envs():
    args = get_args()
    
    os.unsetenv("NCCL_IB_DISABLE")
    os.unsetenv("NCCL_IBEXT_DISABLE")
    os.unsetenv("NCCL_SOCKET_IFNAME")
    os.unsetenv("NCCL_IB_HCA")
    os.unsetenv("NCCL_NET_GDR_LEVEL")
    os.unsetenv("NCCL_NET")
    os.unsetenv("NCCL_IB_GID_INDEX")
    
    if args.use_hetnet:
        os.environ["NCCL_NET"]=nccl_config.NET_IB
        os.environ["NCCL_IB_DISABLE"]=nccl_config.ENABLE
        os.environ["NCCL_IBEXT_DISABLE"]=nccl_config.ENABLE
        os.environ["NCCL_SOCKET_IFNAME"]=nccl_config.SOCKET_IFNAME
        os.environ["NCCL_IB_HCA"]=nccl_config.ROCE_HCA
        os.environ["NCCL_IB_GID_INDEX"]=nccl_config.IB_GID_INDEX
    else:
        os.environ["NCCL_IB_DISABLE"]=nccl_config.DISABLE
        os.environ["NCCL_IBEXT_DISABLE"]=nccl_config.DISABLE
        os.environ["NCCL_SOCKET_IFNAME"]=nccl_config.SOCKET_IFNAME
        os.environ["NCCL_NET"]=nccl_config.NET_Socket


def init_nccl_net(group):
    temp = torch.ones(1, device="cuda")
    torch.distributed.all_reduce(temp, group = group)
    torch.cuda.synchronize()


def new_nccl_socket_group(ranks):
    set_nccl_socket_envs()
    group = torch.distributed.new_group(ranks, backend = "nccl",timeout=datetime.timedelta(seconds=1800))
    init_nccl_net(group=group)
    return group


def new_nccl_ib_group(ranks):
    set_nccl_ib_envs()
    group = torch.distributed.new_group(ranks, backend = "nccl")
    init_nccl_net(group=group)
    return group


def new_nccl_roce_group(ranks):
    set_nccl_roce_envs()
    group = torch.distributed.new_group(ranks, backend = "nccl")
    init_nccl_net(group=group)
    return group


def new_process_group(ranks, backend = "nccl"):
    """
    This function creates process groups.
    In addition to simply creating the process groups, it initializes NCCL
    for hybrid IB/Socket network like in the following diagram:
                            ____________
      [GPU Node 0]---TCP---|            |---TCP---[GPU Node 2]
         |                 |            |            |
         |                 |            |            |
        IB                 | IP Network |           IB
         |                 |            |            |
         |                 |            |            |
      [GPU Node 1]---TCP---|____________|---TCP---[GPU Node 3]
    If an environment variable NUM_GPUS_PER_IB_BLOCK is defined it looks up the ranks
    and determines whether the list of ranks belong to the same computational block where
    GPUs nodes are interconnected via IB type of connection or not.
    If all ranks are in the same block, the process group will use NCCL_NET=IB for
    communication, otherwise it will use NCCL_NET=Socket.
    If NCCL_NET=Socket is ever to be used, the user must set NCCL_SOCKET_IFNAME.
    Additionally, it is recommended to set NCCL_SOCKET_NTHREADS and
    NCCL_NSOCKS_PERTHREAD before running the job.
    See: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
    for more info
    The core assumption for this functionality is that the ranks are evenly divided
    into IB blocks and all these IB blocks are of the same size.
    """
    # Get the size of IB block
    compute_block_size = os.getenv("NUM_GPUS_PER_IB_BLOCK")
    num_ib_block = int(os.getenv("NUM_IB_BLOCK"))
    
    if backend == "nccl" and compute_block_size is not None:
        # Determine whether ranks in the list belong to the same IB block or not
        # and create a process group with appropriate NCCL_NET
        compute_block_size = int(compute_block_size)
        blocks = [rank // compute_block_size for rank in ranks]
        # ib node as master node (2ib a& 2roce, symmetrical)
        # use_ib = all(block == blocks[0] for block in blocks) and blocks[0] == 0
        # use_roce = all(block == blocks[0] for block in blocks) and blocks[0] >= 1
        
        # ib node as master node (4ib a& 2roce -> 2ib & 2ib & 2roce, asymmetrical, num_ib_block=2)
        use_ib = all(block == blocks[0] for block in blocks) and blocks[0] < num_ib_block
        use_roce = all(block == blocks[0] for block in blocks) and blocks[0] >= num_ib_block
        if use_ib:
            print_rank_0("use_ib")
            print_rank_0(list(ranks))
            print_rank_last("use_ib")
            print_rank_last(list(ranks))
            return new_nccl_ib_group(ranks)
        
        elif use_roce:
            print_rank_0("use_roce")
            print_rank_0(list(ranks))
            print_rank_last("use_roce")
            print_rank_last(list(ranks))
            return new_nccl_roce_group(ranks)
    
        else:
            print_rank_0("use_socket")
            print_rank_0(list(ranks))
            print_rank_last("use_socket")
            print_rank_last(list(ranks))
            return new_nccl_socket_group(ranks)
            
            
    else:
        return torch.distributed.new_group(ranks, backend=backend)
