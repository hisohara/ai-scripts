# Index 
- [RCCL from PyTorch](#rccl-from-pytorch)
- [RCCL Tests with MPI (AAC8 Kubernetes)](#rccl-tests-with-mpi-aac8-kubernetes)
- [RCCL Tests with MPI (AAC10 MI300X Slurm)](#rccl-tests-with-mpi-aac10-mi300x-slurm)
- [RCCL Tests with MPI (AAC11 MI325X Slurm)](#rccl-tests-with-mpi-aac11-mi325x-slurm)
- [RCCL Tests with MPI (AAC14 MI355X with Pollara 400 Slurm)](#rccl-tests-with-mpi-aac14-mi355x-with-pollara-400-slurm)
- [linux-rdma/perftest (AAC14 MI355X with Pollara 400 Slurm)](#linux-rdmaperftest-aac14-mi355x-with-pollara-400-slurm)

# RCCL
## RCCL from PyTorch
Refer to [PyTorch: Start Locally](https://pytorch.org/get-started/locally/)
for the installation of pytorch. My preference is the installation with venv.

```bash
# Single node
$ export NCCL_DEBUG=INFO
$ torchrun --nproc_per_node=8 rccl_allreduce.py

# Two nodes
## Node0
$ torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --rdzv-id=allreduce --rdzv-backend=c10d --rdzv-endpoint=<Node0 IP>:29500 rccl_allreduce.py
## Node1
$ torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --rdzv-id=allreduce --rdzv-backend=c10d --rdzv-endpoint=<Node0 IP>:29500 rccl_allreduce.py
```

## RCCL Tests with MPI (AAC8 Kubernetes)
```bash
# Build UCX and OpenMPI
$ ./build_mpi.sh

# Build rccl-tests
$ ./build_rccl-tests.sh

# For RUN
MPI_DIR=<MPI install dir>
export PATH=$MPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH

$ mpirun -np 8 --mca pml ob1 --mca btl self,vader ./rccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
<snip>
# nThread 1 nGpus 1 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop:a7809b3
# Using devices
#  Rank  0 Group  0 Pid   7085 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  0 [0000:05:00] AMD Instinct MI300X
#  Rank  1 Group  0 Pid   7086 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  1 [0000:25:00] AMD Instinct MI300X
#  Rank  2 Group  0 Pid   7087 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  2 [0000:45:00] AMD Instinct MI300X
#  Rank  3 Group  0 Pid   7088 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  3 [0000:65:00] AMD Instinct MI300X
#  Rank  4 Group  0 Pid   7089 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  4 [0000:85:00] AMD Instinct MI300X
#  Rank  5 Group  0 Pid   7090 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  5 [0000:a5:00] AMD Instinct MI300X
#  Rank  6 Group  0 Pid   7091 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  6 [0000:c5:00] AMD Instinct MI300X
#  Rank  7 Group  0 Pid   7092 on 1942ef50-8240-449e-82ca-4ce7993f70b4-78bfb5d4f4-4z9xh device  7 [0000:e5:00] AMD Instinct MI300X
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    19.28    0.00    0.00      0    17.24    0.00    0.00      0
          16             4     float     sum      -1    16.32    0.00    0.00      0    16.53    0.00    0.00      0
          32             8     float     sum      -1    17.10    0.00    0.00      0    17.33    0.00    0.00      0
          64            16     float     sum      -1    18.59    0.00    0.01      0    20.95    0.00    0.01      0
         128            32     float     sum      -1    23.05    0.01    0.01      0    24.40    0.01    0.01      0
         256            64     float     sum      -1    23.36    0.01    0.02      0    23.14    0.01    0.02      0
         512           128     float     sum      -1    23.21    0.02    0.04      0    23.42    0.02    0.04      0
        1024           256     float     sum      -1    13.67    0.07    0.13      0    14.87    0.07    0.12      0
        2048           512     float     sum      -1    13.67    0.15    0.26      0    13.45    0.15    0.27      0
        4096          1024     float     sum      -1    13.72    0.30    0.52      0    13.33    0.31    0.54      0
        8192          2048     float     sum      -1    13.76    0.60    1.04      0    13.42    0.61    1.07      0
       16384          4096     float     sum      -1    15.47    1.06    1.85      0    14.11    1.16    2.03      0
       32768          8192     float     sum      -1    13.94    2.35    4.11      0    13.59    2.41    4.22      0
       65536         16384     float     sum      -1    14.25    4.60    8.05      0    13.82    4.74    8.30      0
      131072         32768     float     sum      -1    14.60    8.98   15.71      0    15.65    8.38   14.66      0
      262144         65536     float     sum      -1    17.46   15.02   26.28      0    16.88   15.53   27.17      0
      524288        131072     float     sum      -1    27.30   19.20   33.60      0    24.75   21.19   37.08      0
     1048576        262144     float     sum      -1    28.39   36.94   64.64      0    28.04   37.40   65.45      0
     2097152        524288     float     sum      -1    36.68   57.17  100.05      0    35.80   58.57  102.50      0
     4194304       1048576     float     sum      -1    52.96   79.20  138.60      0    54.34   77.18  135.07      0
     8388608       2097152     float     sum      -1    84.78   98.95  173.16      0    87.87   95.46  167.06      0
    16777216       4194304     float     sum      -1    149.9  111.92  195.85      0    156.3  107.33  187.83      0
    33554432       8388608     float     sum      -1    246.3  136.25  238.45      0    258.9  129.59  226.78      0
    67108864      16777216     float     sum      -1    432.5  155.15  271.51      0    443.0  151.49  265.11      0
   134217728      33554432     float     sum      -1    800.3  167.72  293.50      0    808.3  166.04  290.58      0
   268435456      67108864     float     sum      -1   1540.4  174.26  304.96      0   1549.8  173.20  303.10      0
   536870912     134217728     float     sum      -1   3027.1  177.35  310.37      0   3040.1  176.59  309.04      0
  1073741824     268435456     float     sum      -1   5990.7  179.24  313.66      0   5996.4  179.06  313.36      0
  2147483648     536870912     float     sum      -1    11940  179.86  314.75      0    11940  179.86  314.75      0
  4294967296    1073741824     float     sum      -1    23841  180.15  315.26      0    23893  179.76  314.58      0
  8589934592    2147483648     float     sum      -1    47617  180.40  315.69      0    47584  180.52  315.91      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 110.464
#
```

## RCCL Tests with MPI (AAC10 MI300X Slurm)
```bash
$ module load rocm-6.4.2/ucx-1.18.0/ompi/5.0.3
$ rdma link
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev ens6np0
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev ens5np0
link mlx5_2/1 state ACTIVE physical_state LINK_UP netdev ens8np0
link mlx5_3/1 state ACTIVE physical_state LINK_UP netdev ens7np0
link mlx5_4/1 state ACTIVE physical_state LINK_UP netdev ens2np0
link mlx5_5/1 state ACTIVE physical_state LINK_UP netdev ens1np0
link mlx5_6/1 state ACTIVE physical_state LINK_UP netdev ens4np0
link mlx5_9/1 state ACTIVE physical_state LINK_UP netdev ens3np0
link mlx5_bond_0/1 state ACTIVE physical_state LINK_UP netdev ens10f0np0

# Single node
$ mpirun -np 8 -H gpu-53:8 --mca pml ucx -x UCX_IB_GID_INDEX=3 -x NCCL_DEBUG=WARN \
-x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=3 -x NCCL_SOCKET_IFNAME=ens -x NCCL_IB_HCA=^mlx5_bond_0 \
-x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1 \
/shared/apps/ubuntu/rocm-6.3.4/rccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

# Two nodes
$ mpirun -np 16 -H gpu-53:8,gpu-56:8 --mca pml ucx -x UCX_IB_GID_INDEX=3 -x NCCL_DEBUG=WARN \
-x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=3 -x NCCL_SOCKET_IFNAME=ens -x NCCL_IB_HCA=^mlx5_bond_0 \
-x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1 \
/shared/apps/ubuntu/rocm-6.3.4/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1
<snip>
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop:448c4c7
# Using devices
#   Rank  0 Pid 3844613 on     gpu-53 device  0 [0000:05:00.0] AMD Instinct MI300X
#   Rank  1 Pid 3844614 on     gpu-53 device  1 [0000:25:00.0] AMD Instinct MI300X
#   Rank  2 Pid 3844615 on     gpu-53 device  2 [0000:45:00.0] AMD Instinct MI300X
#   Rank  3 Pid 3844616 on     gpu-53 device  3 [0000:65:00.0] AMD Instinct MI300X
#   Rank  4 Pid 3844617 on     gpu-53 device  4 [0000:85:00.0] AMD Instinct MI300X
#   Rank  5 Pid 3844618 on     gpu-53 device  5 [0000:a5:00.0] AMD Instinct MI300X
#   Rank  6 Pid 3844619 on     gpu-53 device  6 [0000:c5:00.0] AMD Instinct MI300X
#   Rank  7 Pid 3844620 on     gpu-53 device  7 [0000:e5:00.0] AMD Instinct MI300X
#   Rank  8 Pid 366643 on     gpu-56 device  0 [0000:05:00.0] AMD Instinct MI300X
#   Rank  9 Pid 366642 on     gpu-56 device  1 [0000:25:00.0] AMD Instinct MI300X
#   Rank 10 Pid 366648 on     gpu-56 device  2 [0000:45:00.0] AMD Instinct MI300X
#   Rank 11 Pid 366650 on     gpu-56 device  3 [0000:65:00.0] AMD Instinct MI300X
#   Rank 12 Pid 366645 on     gpu-56 device  4 [0000:85:00.0] AMD Instinct MI300X
#   Rank 13 Pid 366646 on     gpu-56 device  5 [0000:a5:00.0] AMD Instinct MI300X
#   Rank 14 Pid 366647 on     gpu-56 device  6 [0000:c5:00.0] AMD Instinct MI300X
#   Rank 15 Pid 366644 on     gpu-56 device  7 [0000:e5:00.0] AMD Instinct MI300X
RCCL version : 2.22.3-HEAD:7d8d67c
HIP version  : 6.4.43484-123eb5128
ROCm version : 6.4.2.0-120-e7d83f5
Hostname     : gpu-53
Librccl path : /shared/apps/ubuntu/opt/rocm-6.4.2/lib/librccl.so.1
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    256.5    0.00    0.00      0    254.7    0.00    0.00      0
          16             4     float     sum      -1    218.7    0.00    0.00      0    218.9    0.00    0.00      0
          32             8     float     sum      -1    218.6    0.00    0.00      0    219.3    0.00    0.00      0
          64            16     float     sum      -1    218.8    0.00    0.00      0    219.2    0.00    0.00      0
         128            32     float     sum      -1    219.8    0.00    0.00      0    219.4    0.00    0.00      0
         256            64     float     sum      -1    219.9    0.00    0.00      0    219.7    0.00    0.00      0
         512           128     float     sum      -1    220.8    0.00    0.00      0    221.9    0.00    0.00      0
        1024           256     float     sum      -1    220.8    0.00    0.01      0    221.0    0.00    0.01      0
        2048           512     float     sum      -1    223.6    0.01    0.02      0    223.7    0.01    0.02      0
        4096          1024     float     sum      -1    226.2    0.02    0.03      0    225.6    0.02    0.03      0
        8192          2048     float     sum      -1    240.5    0.03    0.06      0    240.3    0.03    0.06      0
       16384          4096     float     sum      -1    238.8    0.07    0.13      0    239.1    0.07    0.13      0
       32768          8192     float     sum      -1    239.9    0.14    0.26      0    240.0    0.14    0.26      0
       65536         16384     float     sum      -1    241.1    0.27    0.51      0    240.9    0.27    0.51      0
      131072         32768     float     sum      -1    242.9    0.54    1.01      0    242.7    0.54    1.01      0
      262144         65536     float     sum      -1    248.8    1.05    1.98      0    248.0    1.06    1.98      0
      524288        131072     float     sum      -1    305.6    1.72    3.22      0    299.3    1.75    3.28      0
     1048576        262144     float     sum      -1    414.8    2.53    4.74      0    413.6    2.54    4.75      0
     2097152        524288     float     sum      -1    318.6    6.58   12.34      0    313.0    6.70   12.56      0
     4194304       1048576     float     sum      -1    337.9   12.41   23.27      0    339.0   12.37   23.20      0
     8388608       2097152     float     sum      -1    402.2   20.85   39.10      0    402.4   20.84   39.08      0
    16777216       4194304     float     sum      -1    497.3   33.74   63.26      0    502.5   33.39   62.60      0
    33554432       8388608     float     sum      -1    590.3   56.84  106.58      0    596.3   56.27  105.51      0
    67108864      16777216     float     sum      -1    821.9   81.65  153.10      0    826.8   81.16  152.18      0
   134217728      33554432     float     sum      -1   1448.8   92.64  173.70      0   1434.0   93.60  175.49      0
   268435456      67108864     float     sum      -1   1713.4  156.67  293.75      0   1728.4  155.31  291.20      0
   536870912     134217728     float     sum      -1   3096.3  173.39  325.10      0   3113.0  172.46  323.36      0
  1073741824     268435456     float     sum      -1   5908.3  181.74  340.75      0   5918.5  181.42  340.16      0
  2147483648     536870912     float     sum      -1    11574  185.55  347.90      0    11593  185.24  347.32      0
  4294967296    1073741824     float     sum      -1    22788  188.48  353.40      0    22796  188.41  353.26      0
  8589934592    2147483648     float     sum      -1    45224  189.94  356.14      0    45242  189.87  356.00      0
 17179869184    4294967296     float     sum      -1    90115  190.64  357.46      0    90127  190.62  357.41      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 92.3318
#
```

## RCCL Tests with MPI (AAC11 MI325X Slurm)
### 1 node all_reduce_perf run on each allocated node
```bash
$ more rccl-test-allreduce-1nodex8.sh
#!/bin/bash

module load rocm-6.4.1/ucx-1.18.0/ompi/5.0.7

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

for node in $nodes; do
        echo "====== $node ======"
        mpirun -np 8 -H $node:8 /shared/apps/ubuntu/rocm-6.4.1/rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1
done

$ ./rccl-test-allreduce-1nodex8.sh
```

### all_reduce_perf run on 1/2/4/8 nodes
```bash
$ more rccl-test-allreduce-multi.sh
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

$ ./rccl-test-allreduce-multi.sh 2>&1 |tee result-allreduce-multi.log
$ egrep "node|4294967296|8589934592|17179869184" result-allreduce-multi.log |grep -v nThread
====== 1-node ======
  4294967296    1073741824     float     sum      -1    23719  181.07  316.88      0    23774  180.66  316.16      0
  8589934592    2147483648     float     sum      -1    47217  181.93  318.37      0    47268  181.73  318.03      0
 17179869184    4294967296     float     sum      -1    94222  182.33  319.08      0    94185  182.40  319.21      0
====== 2-node ======
  4294967296    1073741824     float     sum      -1    23192  185.19  347.23      0    23209  185.06  346.99      0
  8589934592    2147483648     float     sum      -1    46123  186.24  349.20      0    46145  186.15  349.03      0
 17179869184    4294967296     float     sum      -1    91973  186.79  350.24      0    91996  186.75  350.15      0
====== 4-node ======
  4294967296    1073741824     float     sum      -1    23904  179.68  348.12      0    23914  179.60  347.98      0
  8589934592    2147483648     float     sum      -1    47563  180.60  349.92      0    47624  180.37  349.47      0
 17179869184    4294967296     float     sum      -1    94942  180.95  350.59      0    94968  180.90  350.50      0
====== 8-node ======
  4294967296    1073741824     float     sum      -1    24572  174.79  344.13      0    24588  174.68  343.89      0
  8589934592    2147483648     float     sum      -1    48321  177.77  349.98      0    48348  177.67  349.79      0
 17179869184    4294967296     float     sum      -1    96405  178.21  350.84      0    96498  178.03  350.50      0
```
## RCCL Tests with MPI (AAC14 MI355X with Pollara 400 Slurm)
AMD ANP (AINIC Network Plugin) is required with RCCL.

### References
- [GitHub: ROCm/amd-anp](https://github.com/ROCm/amd-anp)
- [AMD AI NIC Pollara 400 Adapter Operations and Troubleshooting User Guide (UG1801)](https://docs.amd.com/r/en-US/ug1801-ai-nic-pollara-400-ops-guide/RCCL-and-ANP-Installation-and-Configuration)

```bash
# Build UCX, OpenMPI, RCCL and ANP
$ ./build_all-anp.sh

# Build rccl-tests linked with built RCCL library
$ ./build_rccl-tests-anp.sh

# Run script
$ more ./rccl-test-allreduce-multi-anp.sh
#!/bin/bash

set -euo pipefail

module load rocm/7.0.1

UCX_INSTALL_DIR=<set your env>
MPI_INSTALL_DIR=<set your env>
RCCL_INSTALL_DIR=<set your env>
ANP_INSTALL_DIR=<set your env>


BINARY_DIR=<set your env>

export PATH=${MPI_INSTALL_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${UCX_INSTALL_DIR}/lib:${MPI_INSTALL_DIR}/lib:${RCCL_INSTALL_DIR}/lib:${ANP_INSTALL_DIR}:$LD_LIBRARY_PATH

echo "==================================="
echo "ANP_HOME_DIR:    $ANP_INSTALL_DIR"
echo "RCCL_HOME_DIR:   $RCCL_INSTALL_DIR"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "==================================="
echo ""

COMMAND=(
    mpirun
    -H mi355-gpu-23:8,mi355-gpu-24:8
    --map-by slot
    --bind-to numa
    --mca btl '^vader,openib'
    --mca pml ob1
    --mca btl_tcp_if_include enp193s0f1np1
    -x NCCL_SOCKET_IFNAME=enp193s0f1np1
    -x IONIC_LOCKFREE=all
    -x NCCL_IB_TC=104
    -x NCCL_IB_FIFO_TC=192
    -x NCCL_IB_GID_INDEX=1
    -x NCCL_GDR_FLUSH_DISABLE=1
    -x NCCL_DEBUG=Version
    -x NCCL_IGNORE_CPU_AFFINITY=1
    -x NCCL_NET_OPTIONAL_RECV_COMPLETION=1
    -x NCCL_PXN_DISABLE=0
    -x NCCL_IB_USE_INLINE=1
    -x NCCL_IB_HCA=rocep121s0:1,rocep9s0:1,rocep105s0:1,rocep25s0:1,rocep249s0:1,rocep137s0:1,rocep233s0:1,rocep153s0:1
    -x NCCL_DMABUF_ENABLE=0
    -x RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
    -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    -x LD_PRELOAD="${ANP_INSTALL_DIR}/librccl-net.so:${RCCL_INSTALL_DIR}/lib/librccl.so.1.0"
    ${BINARY_DIR}/all_reduce_perf
    -g 1
    -f 2
    -b 8
    -e 16G
    -n 20
    -N 2
)

printf "%s " "${COMMAND[@]}"
echo ""

# Execute the command stored in the array
"${COMMAND[@]}"
```
### Result on AAC14 with 2-node
```bash
rccl-tests: Version develop:5272cd1
# Using devices
#  Rank  0 Group  0 Pid 2908511 on mi355-gpu-23 device  0 [0000:75:00]
#  Rank  1 Group  0 Pid 2908512 on mi355-gpu-23 device  1 [0000:05:00]
#  Rank  2 Group  0 Pid 2908513 on mi355-gpu-23 device  2 [0000:65:00]
#  Rank  3 Group  0 Pid 2908514 on mi355-gpu-23 device  3 [0000:15:00]
#  Rank  4 Group  0 Pid 2908515 on mi355-gpu-23 device  4 [0000:f5:00]
#  Rank  5 Group  0 Pid 2908516 on mi355-gpu-23 device  5 [0000:85:00]
#  Rank  6 Group  0 Pid 2908517 on mi355-gpu-23 device  6 [0000:e5:00]
#  Rank  7 Group  0 Pid 2908518 on mi355-gpu-23 device  7 [0000:95:00]
#  Rank  8 Group  0 Pid 1167745 on mi355-gpu-24 device  0 [0000:75:00]
#  Rank  9 Group  0 Pid 1167740 on mi355-gpu-24 device  1 [0000:05:00]
#  Rank 10 Group  0 Pid 1167742 on mi355-gpu-24 device  2 [0000:65:00]
#  Rank 11 Group  0 Pid 1167744 on mi355-gpu-24 device  3 [0000:15:00]
#  Rank 12 Group  0 Pid 1167743 on mi355-gpu-24 device  4 [0000:f5:00]
#  Rank 13 Group  0 Pid 1167741 on mi355-gpu-24 device  5 [0000:85:00]
#  Rank 14 Group  0 Pid 1167746 on mi355-gpu-24 device  6 [0000:e5:00]
#  Rank 15 Group  0 Pid 1167747 on mi355-gpu-24 device  7 [0000:95:00]
RCCL version : 2.26.6-HEAD:6ade506
HIP version  : 7.0.51831-a3e329ad8
ROCm version : 7.0.1.0-42-9428210
Hostname     : mi355-gpu-23
Librccl path : /shared/amdgpu/home/hisaki_ohara_7kq/Projects/ai-scripts/training/01_rccl-smoketest/RCCL-ANP.1/rccl-6ade506/build/release/build/lib/librccl.so.1.0
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
# Testing 1 cycle.
           8             2     float     sum      -1    41.16    0.00    0.00      0    39.80    0.00    0.00      0
          16             4     float     sum      -1    40.09    0.00    0.00      0    39.52    0.00    0.00      0
          32             8     float     sum      -1    39.34    0.00    0.00      0    40.04    0.00    0.00      0
          64            16     float     sum      -1    40.38    0.00    0.00      0    43.36    0.00    0.00      0
         128            32     float     sum      -1    41.41    0.00    0.01      0    41.59    0.00    0.01      0
         256            64     float     sum      -1    43.11    0.01    0.01      0    43.29    0.01    0.01      0
         512           128     float     sum      -1    44.18    0.01    0.02      0    44.31    0.01    0.02      0
        1024           256     float     sum      -1    47.20    0.02    0.04      0    47.28    0.02    0.04      0
        2048           512     float     sum      -1    50.81    0.04    0.08      0    50.82    0.04    0.08      0
        4096          1024     float     sum      -1    55.28    0.07    0.14      0    52.19    0.08    0.15      0
        8192          2048     float     sum      -1    55.27    0.15    0.28      0    53.00    0.15    0.29      0
       16384          4096     float     sum      -1    56.25    0.29    0.55      0    54.10    0.30    0.57      0
       32768          8192     float     sum      -1    56.51    0.58    1.09      0    56.03    0.58    1.10      0
       65536         16384     float     sum      -1    59.17    1.11    2.08      0    59.74    1.10    2.06      0
      131072         32768     float     sum      -1    65.97    1.99    3.73      0    66.40    1.97    3.70      0
      262144         65536     float     sum      -1    67.22    3.90    7.31      0    67.44    3.89    7.29      0
      524288        131072     float     sum      -1    69.78    7.51   14.09      0    68.90    7.61   14.27      0
     1048576        262144     float     sum      -1    81.37   12.89   24.16      0    82.36   12.73   23.87      0
     2097152        524288     float     sum      -1    98.50   21.29   39.92      0    99.08   21.17   39.68      0
     4194304       1048576     float     sum      -1    118.6   35.36   66.30      0    119.5   35.09   65.79      0
     8388608       2097152     float     sum      -1    166.5   50.39   94.49      0    164.6   50.95   95.54      0
    16777216       4194304     float     sum      -1    261.6   64.12  120.23      0    259.3   64.70  121.30      0
    33554432       8388608     float     sum      -1    391.8   85.65  160.59      0    391.4   85.74  160.76      0
    67108864      16777216     float     sum      -1    614.3  109.25  204.84      0    612.6  109.55  205.40      0
   134217728      33554432     float     sum      -1   1041.8  128.84  241.57      0   1043.5  128.63  241.17      0
   268435456      67108864     float     sum      -1   1652.0  162.49  304.67      0   1604.5  167.30  313.68      0
   536870912     134217728     float     sum      -1   2880.5  186.38  349.47      0   2865.4  187.36  351.31      0
  1073741824     268435456     float     sum      -1   5525.0  194.34  364.39      0   5525.8  194.31  364.34      0
  2147483648     536870912     float     sum      -1    10989  195.41  366.40      0    10990  195.41  366.39      0
  4294967296    1073741824     float     sum      -1    21870  196.38  368.22      0    21868  196.41  368.26      0
  8589934592    2147483648     float     sum      -1    43643  196.82  369.04      0    43639  196.84  369.08      0
```

# linux-rdma/perftest (AAC14 MI355X with Pollara 400 Slurm)
## Confirma mapping of NIC device with GPU ID
```bash
hisaki_ohara_7kq@gpu-6:~$ lstopo-no-graphics | grep -iE 'HostBridge|ProcessingAccelerator|bnxt'
    HostBridge
                PCI 05:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re1"
    HostBridge
                PCI 15:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re3"
    HostBridge
    HostBridge
    HostBridge
    HostBridge
                PCI 65:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re2"
    HostBridge
                PCI 75:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re0"
    HostBridge
                PCI 85:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re5"
    HostBridge
                PCI 95:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re8"
    HostBridge
    HostBridge
          OpenFabrics "bnxt_re6"
    HostBridge
    HostBridge
                PCI e5:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re7"
    HostBridge
                PCI f5:00.0 (ProcessingAccelerator)
              OpenFabrics "bnxt_re4"

hisaki_ohara_7kq@gpu-6:~$ amd-smi list|egrep "GPU|BDF"
GPU: 0
    BDF: 0000:05:00.0
GPU: 1
    BDF: 0000:15:00.0
GPU: 2
    BDF: 0000:65:00.0
GPU: 3
    BDF: 0000:75:00.0
GPU: 4
    BDF: 0000:85:00.0
GPU: 5
    BDF: 0000:95:00.0
GPU: 6
    BDF: 0000:e5:00.0
GPU: 7
    BDF: 0000:f5:00.0
```
For the case of above (on AAC11, the same way for AINIC), the mapping is as like follows.

| NIC rdma ID | GPU ID |
| --- | --- |
| bnxt_re0 | 3 |
| bnxt_re1 | 0 |
| bnxt_re2 | 2 |
| bnxt_re3 | 1 |
| bnxt_re4 | 7 |
| bnxt_re5 | 4 |
| bnxt_re7 | 6 |
| bnxt_re8 | 5 |

## How to build
```bash
$ git clone https://github.com/linux-rdma/perftest.git
$ cd perftest
$ ./autogen.sh
$ ./configure --prefix=$PWD/install --enable-rocm --with-rocm=/opt/rocm
``` 

## Run script
Adjust `rocm_dev` to reflect above mapping.

```bash
#!/usr/bin/bash

set -x

server="gpu-6" # change this
client="gpu-12" # change this

node=(0 0 0 0 1 1 1 1)
path_to_perftest=<dir to your perftest>
rdmadev=(bnxt_re0 bnxt_re1 bnxt_re2 bnxt_re3 bnxt_re4 bnxt_re5 bnxt_re7 bnxt_re8) # !!! adjust this according to your rdma (openfabrics) NIC device name
rocm_dev=(3 0 2 1 7 4 6 5) # adjust this to reflect what GPU ROCm ID aligns with the rdma (openfabrics) NIC device

num_dev=${#rdmadev[@]}

# bandwidth tests unidirectional
for benchmark in ib_read_bw ib_write_bw ib_send_bw; do
    for i in $(seq 0 $((num_dev - 1))); do
        printf "deviceinfo -- rdmadev: %s,\tlocal_ipaddr: %s,\trocm_dev: %s\n" ${rdmadev[i]} ${server} ${rocm_dev[i]}

        # H2H bandwidth
        killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw
        ${path_to_perftest}/perftest/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits -F -a &
        ssh amd@${client} "killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw"
        ssh amd@${client} "${path_to_perftest}/perftest/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits -F -a ${server}" 2>&1 | tee ${benchmark}_h2h_${rdmadev[i]}_unidi.log

        # D2D bandwidth
        killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw
        ${path_to_perftest}/perftest/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits --use_rocm=${rocm_dev[i]} -F -a &
        ssh amd@${client} "killall ib_send_lat ib_read_lat ib_write_lat ib_send_bw ib_read_bw ib_write_bw"
        ssh amd@${client} "${path_to_perftest}/perftest/install/bin/${benchmark} -d ${rdmadev[i]} -x 3 -q 2 --report_gbits --use_rocm=${rocm_dev[i]} -F -a ${server}" 2>&1 | tee ${benchmark}_d2d_${rdmadev[i]}_rocm${rocm_dev[i]}_unidi.log

    done
done
```




