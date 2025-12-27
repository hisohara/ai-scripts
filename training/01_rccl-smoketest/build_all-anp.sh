#!/bin/bash

set -euo pipefail

module load rocm/7.0.1

WORKDIR=$PWD

UCX_VERSION="v1.18.0"
MPI_VERSION="5.0.7"
RCCL_VERSION="6ade506"
ANP_VERSION="v1.1.0-5"

RCCL_COMMIT="6ade5065b4c0e6338f69778dcd7d477ef5bfcbcd"

UCX_DIR="ucx-${UCX_VERSION}"
MPI_TARBALL="openmpi-${MPI_VERSION}.tar.gz"
MPI_DOWNLOAD_URL="https://download.open-mpi.org/release/open-mpi/v$(echo "${MPI_VERSION}" | cut -d. -f1-2)/openmpi-${MPI_VERSION}.tar.gz"
MPI_DIR="ompi-${MPI_VERSION}"
RCCL_DIR="rccl-${RCCL_VERSION}"
ANP_DIR="anp-${ANP_VERSION}"

# Install UCX
cd "${WORKDIR}"
git clone https://github.com/openucx/ucx.git -b ${UCX_VERSION} ${UCX_DIR}
cd "${UCX_DIR}"
./autogen.sh
./configure --prefix="${WORKDIR}/${UCX_DIR}/install" --with-rocm="${ROCM_PATH}" --without-knem --without-cuda
make -j16
make install
UCX_INSTALL_DIR=${WORKDIR}/${UCX_DIR}/install
echo "UCX ${UCX_VERSION} built and installed successfully on ${UCX_INSTALL_DIR}"

# Install OpenMPI
cd ${WORKDIR}
wget $MPI_DOWNLOAD_URL
mkdir -p "${MPI_DIR}"
tar -zxf "${MPI_TARBALL}" -C "${MPI_DIR}" --strip-components=1
cd "${MPI_DIR}"
./configure --prefix="${WORKDIR}/${MPI_DIR}/install" --with-ucx="${UCX_INSTALL_DIR}" --with-hwloc --disable-oshmem --disable-mpi-fortran
make -j16
make install
MPI_INSTALL_DIR=${WORKDIR}/${MPI_DIR}/install
echo "OpenMPI ${MPI_VERSION} built and installed successfully on ${MPI_INSTALL_DIR}"

# Install RCCL
cd "${WORKDIR}"
git clone https://github.com/ROCm/rccl.git ${RCCL_DIR}
cd "${RCCL_DIR}"
git checkout $RCCL_COMMIT
./install.sh -l --prefix build/ --disable-msccl-kernel
RCCL_INSTALL_DIR=${WORKDIR}/${RCCL_DIR}/build/release/build
echo "RCCL ${RCCL_VERSION} built and installed successfully on ${RCCL_INSTALL_DIR}"

# Build amd-anp
RCCL_HOME="${WORKDIR}/${RCCL_DIR}"
MPI_INCLUDE="${WORKDIR}/${MPI_DIR}/install/include"
MPI_LIB_PATH="${WORKDIR}/${MPI_DIR}/install/lib"
cd ${WORKDIR}
#apt update && apt install -y libfmt-dev
git clone https://github.com/rocm/amd-anp.git -b ${ANP_VERSION} ${ANP_DIR}
cd "${ANP_DIR}"
sed -i.bak '70d;71d' Makefile
make RCCL_HOME=$RCCL_HOME MPI_INCLUDE=$MPI_INCLUDE MPI_LIB_PATH=$MPI_LIB_PATH CC=`which hipcc`
ANP_INSTALL_DIR=${WORKDIR}/${ANP_DIR}/build
echo "ANP ${ANP_VERSION} built and installed successfully on ${ANP_INSTALL_DIR}"

# Summarize directories
echo
echo "UCX_INSTALL_DIR=${UCX_INSTALL_DIR}"
echo "MPI_INSTALL_DIR=${MPI_INSTALL_DIR}"
echo "RCCL_INSTALL_DIR=${RCCL_INSTALL_DIR}"
echo "ANP_INSTALL_DIR=${ANP_INSTALL_DIR}"
