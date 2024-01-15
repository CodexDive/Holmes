#!/usr/bin/expect -f
set timeout 500
for {set index 0} {$index < 1} {incr index} {
    spawn ssh ji-jupyter-7211636687642796032-worker-${index}.ji-jupyter-7211636687642796032
    expect "*root*"
    send "cd /gpfs/gpt3/code/Holmes\r"
    expect "*root*"
    send "nohup bash ./examples/pretrain_gpt_2_nodes.sh ${index} >/home/torch2_2nodes_v2.log 2 >& 1 &\r"
    expect "*"
    send "exit\r"
}



