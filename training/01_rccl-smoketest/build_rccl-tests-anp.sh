#!/bin/bash

set -euo pipefail

module load rocm/7.0.1

UCX_INSTALL_DIR=<add your env>
MPI_INSTALL_DIR=<add your env>
RCCL_INSTALL_DIR=<add your env>
ANP_INSTALL_DIR=<add your env>

export PATH=${MPI_INSTALL_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${UCX_INSTALL_DIR}/lib:${MPI_INSTALL_DIR}/lib:${RCCL_INSTALL_DIR}/lib:$LD_LIBRARY_PATH

git clone https://github.com/ROCm/rccl-tests
cd rccl-tests
make MPI=1 MPI_HOME=${MPI_INSTALL_DIR} NCCL_HOME=${RCCL_INSTALL_DIR} CUSTOM_RCCL_LIB=${RCCL_INSTALL_DIR}/lib/librccl.so -j16

ldd build/all_reduce_perf
