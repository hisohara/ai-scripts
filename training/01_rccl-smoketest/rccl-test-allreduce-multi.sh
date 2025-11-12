#!/bin/bash

module load rocm-6.4.1/ucx-1.18.0/ompi/5.0.7

list_1=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1 | paste -sd, - | sed 's/,/:8,/g;s/$/:8/')
list_2=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 2 | paste -sd, - | sed 's/,/:8,/g;s/$/:8/')
list_4=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 4 | paste -sd, - | sed 's/,/:8,/g;s/$/:8/')
list_8=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 8 | paste -sd, - | sed 's/,/:8,/g;s/$/:8/')

echo "====== 1-node ======"
mpirun -np 8 -H $list_1 --mca pml ucx --mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=3 \
	-x NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x UCX_NET_DEVICES=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x NCCL_SOCKET_IFNAME=enp49s0f1np1 \
	-x NCCL_ALGO=Ring /shared/apps/ubuntu/rocm-6.4.1/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1

echo "====== 2-node ======"
mpirun -np 16 -H $list_2 --mca pml ucx --mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=3 \
	-x NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x UCX_NET_DEVICES=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x NCCL_SOCKET_IFNAME=enp49s0f1np1 \
	-x NCCL_ALGO=Ring /shared/apps/ubuntu/rocm-6.4.1/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1

echo "====== 4-node ======"
mpirun -np 32 -H $list_4 --mca pml ucx --mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=3 \
	-x NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x UCX_NET_DEVICES=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x NCCL_SOCKET_IFNAME=enp49s0f1np1 \
	-x NCCL_ALGO=Ring /shared/apps/ubuntu/rocm-6.4.1/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1


echo "====== 8-node ======"
mpirun -np 64 -H $list_8 --mca pml ucx --mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=3 \
	-x NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x UCX_NET_DEVICES=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 \
	-x NCCL_SOCKET_IFNAME=enp49s0f1np1 \
	-x NCCL_ALGO=Ring /shared/apps/ubuntu/rocm-6.4.1/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1

