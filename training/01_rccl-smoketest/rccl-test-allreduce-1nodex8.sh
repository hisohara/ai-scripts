#!/bin/bash

module load rocm-6.4.1/ucx-1.18.0/ompi/5.0.7

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

for node in $nodes; do
	echo "====== $node ======"
	mpirun -np 8 -H $node:8 /shared/apps/ubuntu/rocm-6.4.1/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1
done
