# RCCL primitive tests
- [RCCL from PyTorch](#rccl-from-pytorch)
- [RCCL Tests with MPI (AAC8 Kubernetes)](#rccl-tests-with-mpi-aac8-kubernetes)
- [RCCL Tests with MPI (AAC10 MI300X Slurm)](#rccl-tests-with-mpi-aac10-mi300x-slurm)
- [RCCL Tests with MPI (AAC11 MI325X Slurm)](#rccl-tests-with-mpi-aac11-mi325x-slurm)

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
