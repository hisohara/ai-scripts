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
