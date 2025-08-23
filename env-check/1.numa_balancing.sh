#!/bin/bash

current_value=$(cat /proc/sys/kernel/numa_balancing)
echo "Current kernel.numa_balancing value: $current_value"

if [ "$current_value" -eq 1 ]; then
    echo "Disabling NUMA balancing..."
    sudo sysctl kernel.numa_balancing=0
else
    echo "No change needed."
fi
