# Primus for distributed training
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

## Examples of outputs
- [Llama2 70B 2-node](run-llama2-70B-2N.log)
- [Llama2 7B 8-node](run-llama2-7B-8N.log)
- [Mixtral 8x7B 8-node](run-mixtral-8x7B-8N.log)
