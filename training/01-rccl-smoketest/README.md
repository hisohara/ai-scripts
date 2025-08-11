# RCCL primitive tests
## RCCL from PyTorch
```bash
export NCCL_DEBUG=INFO

torchrun --nproc_per_node=8 rccl_allreduce.py
```

## rccl-tests with MPI
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
```
### Typical output
```bash
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
