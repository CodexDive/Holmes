#!/bin/bash
nohup bash examples/pretrain_gpt_2_nodes.sh 0 ib >/home/torch2_2nodes_v2.log 2 >& 1 &
expect examples/create_train_job_2n.sh
