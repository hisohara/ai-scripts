#!/bin/bash

MPI_DIR=/home/aac/Projects/OpenMPI508-rocm640-ucx118
RCCL_DIR="${ROCM_PATH:-/opt/rocm}"

export PATH=$MPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH

git clone https://github.com/ROCm/rccl-tests
cd rccl-tests
make MPI=1 MPI_HOME=${MPI_DIR} NCCL_HOME=${RCCL_DIR} -j

