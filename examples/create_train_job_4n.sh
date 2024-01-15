#!/usr/bin/expect -f
set timeout 500
for {set index 0} {$index < 3} {incr index} {
    spawn ssh ji-aitrain-7208681403089133568-worker-${index}.ji-aitrain-7208681403089133568
    expect "*root*"
    send "cd /gpfs/gpt3/code/Holmes\r"
    expect "*root*"
    send "nohup bash ./examples/pretrain_gpt_4_nodes.sh ${index} >/home/torch2_4nodes_v1.log 2 >& 1 &\r"
    expect "*"
    send "exit\r"
}



