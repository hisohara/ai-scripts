# Primus for distributed training
## Index
- [Primus evaluation on AAC11 (MI325X)](#primus-evaluation-on-aac11-mi325x)
- [Primus evaluation on AAC14 (MI355X)](#primus-evaluation-on-aac14-mi355x)
## References
- [Training a model with Primus and Megatron-LM](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html)
- [GitHub: AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)
- [Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)
- [An Introduction to Primus-Turbo: A Library for Accelerating Transformer Models on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus-large-models/README.html)

## Primus evaluation on AAC11 (MI325X)
```bash
$ git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
$ cd Primus
$ git checkout e16b27b
$ git submodule
+847781764fe468c90caec16309deded245c1022c third_party/Megatron-LM (25.04-alpha.rc1-1241-g847781764)
+99c0cb28f615d99290273afa1da01fd72f01f1a5 third_party/torchtitan (v0.1.0-184-g99c0cb28)
$ git submodule update --init --recursive
Submodule path 'third_party/Megatron-LM': checked out '5a676b361810f3037c5dd78ef891bbc1abc4a21b'
Submodule path 'third_party/torchtitan': checked out '4c4388f65047393acd91deb3f7a31d673d41701f'
$ git submodule
 5a676b361810f3037c5dd78ef891bbc1abc4a21b third_party/Megatron-LM (25.04-alpha.rc1-426-g5a676b361)
 4c4388f65047393acd91deb3f7a31d673d41701f third_party/torchtitan (v0.1.0~27)
$ mkdir data
```

Download Broadcom software from
https://www.broadcom.com/products/ethernet-connectivity/network-adapters/p1400g
Version 231.2.63.0 is used for this evaluation and put `libbnxt_re-231.0.162.0.tar.gz` into your environment.

Modify run scripts per the environment on AAC11.
```bash
$ git diff
diff --git a/examples/run_pretrain.sh b/examples/run_pretrain.sh
index e64bbd7..1b714c2 100755
--- a/examples/run_pretrain.sh
+++ b/examples/run_pretrain.sh
@@ -255,6 +255,8 @@ export PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE}
 if [[ "$REBUILD_BNXT" == "1" && -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
     LOG_INFO "Rebuilding bnxt from $PATH_TO_BNXT_TAR_PACKAGE ..." && \
     tar xzf "${PATH_TO_BNXT_TAR_PACKAGE}" -C /tmp/ && \
+    apt -y install linux-headers-"$(uname -r)" libelf-dev && \
+    apt -y install gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace && \
     mv /tmp/libbnxt_re-* /tmp/libbnxt && \
     mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox && \
     cd /tmp/libbnxt/ && sh ./autogen.sh && ./configure && \
diff --git a/examples/run_slurm_pretrain.sh b/examples/run_slurm_pretrain.sh
index fd2f606..56c25cc 100755
--- a/examples/run_slurm_pretrain.sh
+++ b/examples/run_slurm_pretrain.sh
@@ -41,6 +41,7 @@ srun -N "${NNODES}" \
      --exclusive \
      --ntasks-per-node=1 \
      --cpus-per-task="${CPUS_PER_TASK:-256}" \
+     --reservation=gpu-14_gpu-4_gpu-11_gpu-6_gpu-12_gpu-28_gpu-29_gpu-30_reservation \
      bash -c "
           readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
           if [ \"\$SLURM_NODEID\" = \"0\" ]; then
```

```bash
$ more RUN.sh
#!/bin/bash

export DOCKER_IMAGE=rocm/primus:v25.9_gfx942
export HF_TOKEN=<YOUR TOKEN>
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 # specify which RDMA interfaces to use for communication
export NCCL_SOCKET_IFNAME=enp49s0f1np1 # your Network Interface
export GLOO_SOCKET_IFNAME=enp49s0f1np1 # your Network Interface
export NCCL_IB_GID_INDEX=3 # Set InfiniBand GID index for NCCL communication. Default is 3 for ROCE
export CPUS_PER_TASK=256
export CLEAN_DOCKER_CONTAINER=1
export REBUILD_BNXT=1
export PATH_TO_BNXT_TAR_PACKAGE=/shared/amdgpu/home/hisaki_ohara_7kq/libbnxt_re-231.0.162.0.tar.gz

## Llama2 70B 2-node
#NNODES=2 \
#EXP=examples/megatron/configs/llama2_70B-pretrain.yaml \
#bash examples/run_slurm_pretrain.sh \
#    --micro_batch_size 10 \
#    --global_batch_size 640 \
#    --recompute_num_layers 80 \
#    --no_fp8_weight_transpose_cache true \
#    --fp8 hybrid

## Llama2 7B 8-node
#NNODES=8 \
#EXP=examples/megatron/configs/llama2_7B-pretrain.yaml \
#bash ./examples/run_slurm_pretrain.sh \
#    --global_batch_size 2048 \
#    --fp8 hybrid

# Mixtral 8x7B 8-node
NNODES=8 \
EXP=examples/megatron/configs/mixtral_8x7B_v0.1-pretrain.yaml \
bash examples/run_slurm_pretrain.sh \
    --micro_batch_size 2 \
    --global_batch_size 256
```

### Examples of outputs
- [Llama2 70B 2-node](run-llama2-70B-2N.log)
- [Llama2 7B 8-node](run-llama2-7B-8N.log)
- [Mixtral 8x7B 8-node](run-mixtral-8x7B-8N.log)

## Primus evaluation on AAC14 (MI355X)
```bash
$ podman pull docker.io/rocm/primus:v26.2
$ podman run -it --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm \
--network=host --ipc=host --group-add keep-groups -e HF_TOKEN=$HF_TOKEN docker.io/rocm/primus:v26.2

$ cd /shared/data
$ tar zxvf ainic_bundle_1.117.5-a-56.tar.gz
$ cd ainic_bundle_1.117.5-a-56/
$ tar zxvf host_sw_pkg.tar.gz
$ cd host_sw_pkg/ionic_driver/src/
$ tar xvf drivers-linux.tar.xz
$ podman cp /shared/data/ainic_bundle_1.117.5-a-56 <CONTAINER_ID>:/tmp
```

### In container
```bash
$ cd /workspace/Primus

# To avoid SEGV, modify examples/run_pretrain.sh
$ diff -u examples/run_pretrain.sh.0 examples/run_pretrain.sh
--- examples/run_pretrain.sh.0  2026-04-23 07:05:17.925132421 +0000
+++ examples/run_pretrain.sh    2026-04-23 07:08:08.036614307 +0000
@@ -223,7 +223,7 @@
         export NCCL_DMABUF_ENABLE=0
         export NCCL_IB_QPS_PER_CONNECTION=1

-        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:${RCCL_HOME_DIR}/build/release:${ANP_HOME_DIR}/build:${MPI_HOME_DIR}/lib:$LD_LIBRARY_PATH
+        export LD_LIBRARY_PATH=${RCCL_HOME_DIR}/build/release:${ANP_HOME_DIR}/build:$LD_LIBRARY_PATH
     fi
     # Check which NCCL net plugin library is present under ${ANP_HOME_DIR}/build and set accordingly
     if [ -f "${ANP_HOME_DIR}/build/librccl-anp.so" ]; then
```

ionic's userland library needs to be install to avoid ABI mismatch.
```bash
$ cd /tmp/ainic_bundle_1.117.5-a-56/host_sw_pkg/ionic_driver/src/drivers-linux/rdma-core/
$ mkdir build; cd build
$ cmake -GNinja -DCMAKE_INSTALL_PREFIX:PATH=/usr -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 $EXTRA_CMAKE_FLAGS ..
$ ninja
$ sudo ninja install
$ ibv_devices
    device                 node GUID
    ------              ----------------
    rocep121s0          069081fffe369f90
    rocep9s0            069081fffe366e28
    rocep105s0          069081fffe36c0a8
    rocep25s0           069081fffe3695b8
    rocep249s0          069081fffe3670f8
    rocep137s0          069081fffe365b98
    rocep233s0          069081fffe366c30
    rocep153s0          069081fffe365b38
    rocep193s0f0        7ec255fffebaf228
    rocep193s0f1        7ec255fffebaf229
$ more RUN.sh
#!/bin/bash

export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml

export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

export NNODES=2                         # Number of nodes (default: 1)
export NODE_RANK=0                      # Current node rank (default: 0)
export GPUS_PER_NODE=8                  # Number of GPUs per node (default: 8)
export MASTER_ADDR=mi355-gpu-22         # Master node address (default: localhost)
export MASTER_PORT=1234                 # Master node port (default: 1234)
export PRIMUS_HIPBLASLT_TUNING_STAGE=0  # HipBLASLt tuning stage: 0/1/2/3 (default: 0)

export NCCL_IB_HCA=rocep105s0:1,rocep121s0:1,rocep137s0:1,rocep153s0:1,rocep233s0:1,rocep249s0:1,rocep25s0:1,rocep9s0:1

export USING_AINIC=1
export ANP_HOME_DIR=/workspace/amd-anp
export RCCL_HOME_DIR=/workspace/rccl
export NCCL_DEBUG=Version

bash examples/run_pretrain.sh
```
In another node, change `NODE_RANK` as 1.
