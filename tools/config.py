import json

with open('config.json', 'r') as file:
    clusters_data = json.load(file)

nccl_config={}
for data in clusters_data:
    cluster_name=data.pop("CLUSTER_NAME")
    nccl_config[cluster_name]=data

