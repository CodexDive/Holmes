#!/bin/bash
cpu_num=$(cat /proc/cpuinfo | grep processor | wc -l)
echo Total $cpu_num CPUs
let cpu_index=cpu_num-1
echo Max index $cpu_index
for i in $(seq 0 $cpu_index)
do
        echo set CPU$i cpufreq governor to $1
        echo $1 > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor
done
for i in `seq 0 7`; do  mx-smi --set-power-mode 1 -i $i;done

