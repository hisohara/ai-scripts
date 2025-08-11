#!/bin/bash

INSTALLDIR=/home/aac/Projects/OpenMPI507-rocm640-ucx118

git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.18.x
./autogen.sh
./configure --prefix=$INSTALLDIR --with-rocm=$ROCM_PATH --without-knem --without-cuda
make -j24
make install
cd ..

#git clone --recursive -b v5.0.7 https://github.com/open-mpi/ompi.git

wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.bz2
tar jxvf openmpi-5.0.8.tar.bz2
cd openmpi-5.0.8
./autogen.pl
./configure --prefix=$INSTALLDIR --with-ucx=$INSTALLDIR --with-hwloc CC=gcc CXX=g++
make -j24
make install
