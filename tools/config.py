import json


with open("config.json", "r") as config_file:
    config = json.load(config_file)
    
    
class NCCL_CONFIG:
    def __init__(self) -> None:
        self.SOCKET_IFNAME = ""
        self.NET_IB = ""
        self.NET_Socket = ""
        self.IB_HCA = ""
        self.ROCE_HCA = ""
        self.NET_GDR_LEVEL = ""
        self.IB_GID_INDEX = ""
        self.DISABLE = ""
        self.ENABLE = ""
    
nccl_config = NCCL_CONFIG()
nccl_config.SOCKET_IFNAME = config["SOCKET_IFNAME"]
nccl_config.NET_IB  =config[ "NET_IB"]
nccl_config.NET_Socket = config["NET_ROCE"]
nccl_config.IB_HCA = config["IB_HCA"]
nccl_config.ROCE_HCA = config["ROCE_HCA"]
nccl_config.NET_GDR_LEVEL = config["NET_GDR_LEVEL"]
nccl_config.IB_GID_INDEX = config["IB_GID_INDEX"]
nccl_config.ENABLE = config["ENABLE"]
nccl_config.DISABLE = config["DISABLE"]
