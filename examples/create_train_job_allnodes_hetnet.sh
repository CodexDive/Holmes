#!/bin/bash
set timeout 500

apt-get install -y iproute2
ssh_config_file="/root/.ssh/config_2"
workers=$(grep "^Host " "$ssh_config_file" | awk '{print $2}')
worker_command="cd /gpfs/gpt3/code/Holmes; nohup bash ./examples/pretrain_gpt_4_nodes.sh "
log=" >/home/torch2_4nodes_roce4.log 2 >& 1 &"
rank_num=0
ini_num=0
interface="bond0"
ip_info=$(ip addr show $interface | grep -w inet)
ip_address=$(echo $ip_info | awk '{print $2}' | awk -F'/' '{print $1}')

for worker in $workers;do
    ip=$(awk -v host="$worker" '/^Host / {p = 0} p && /^ *HostName / {print $2} host == $2 {p = 1}' "$ssh_config_file")
    if [ "$ip" == "$ip_address" ]; then
        echo $ini_num
        echo $worker
        ssh -f $worker "$worker_command$ini_num$log"
    else
        ((rank_num++))
        echo $rank_num
        echo $worker
        ssh -f $worker "$worker_command$rank_num$log"
    fi
done