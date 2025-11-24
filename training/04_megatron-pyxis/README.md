# Megatron-LM for distributed training
## References
- [Training a model with Megatron-LM on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html)
- [Training a model with Primus and Megatron-LM](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html)
- [GitHub: ROCm/Megatron-LM](https://github.com/ROCm/Megatron-LM)
- [GitHub: AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)
- [AAC: How to use enroot](https://aac.amd.com/help/bare-metal/how-to-use-enroot/)

## Megatron-LM evaluation on AAC11 (MI325X)
Pyxis/Enroot is used for this evaluation.

Note that `rocm/megatron-lm` Docker Hub registry will be deprecated soon by replacing with `rocm/primus`. In fact, the latest `rocm/megatron-lm` image is identical as `rocm/primus`.
```bash
$ podman images --digests
REPOSITORY                  TAG                DIGEST                                                                   IMAGE ID      CREATED      SIZE
docker.io/rocm/megatron-lm  v25.9_gfx942       sha256:df6ab8f45b4b9ceb100fb24e19b2019a364e351ee3b324dbe54466a1d67f8357  19d98d384696  5 weeks ago  64.4 GB
docker.io/rocm/primus       v25.9_gfx942       sha256:df6ab8f45b4b9ceb100fb24e19b2019a364e351ee3b324dbe54466a1d67f8357  19d98d384696  5 weeks ago  64.4 GB
```

### Preparation of Enroot image
Please prepare Broadcom userland library file and put onto the working directory.
```bash
$ more Dockerfile.megatron-bnxt
ARG BASE_IMAGE=docker.io/rocm/megatron-lm:v25.9_gfx942
FROM ${BASE_IMAGE}

COPY libbnxt_re-231.0.162.0.tar.gz /tmp

# Install libbnxt
RUN apt update \
    && mkdir -p /tmp/libbnxt \
    && tar xzf /tmp/libbnxt_re-231.0.162.0.tar.gz -C /tmp/libbnxt --strip-components=1 \
    && apt install -y linux-headers-6.5.0-45-generic libelf-dev \
    && apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace \
    && mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox \
    && cd /tmp/libbnxt/ \
    && sh ./autogen.sh \
    && ./configure \
    && make -C /tmp/libbnxt clean all install \
    && echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf \
    && ldconfig \
    && cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/

# Megatron-LM backward compatibility setup
RUN cd /workspace/Megatron-LM/ \
    && pip uninstall megatron-core \
    && pip install -e .
$ podman build . -f Dockerfile.megatron-bnxt -t rocm/megatron-lm:v25.9_gfx942-bnxt
$ podman images
REPOSITORY                  TAG                IMAGE ID      CREATED      SIZE
localhost/rocm/megatron-lm  v25.9_gfx942-bnxt  e5cc829112a7  2 days ago   64.7 GB
docker.io/rocm/megatron-lm  v25.9_gfx942       19d98d384696  5 weeks ago  64.4 GB
$ enroot import podman://rocm/megatron-lm:v25.9_gfx942-bnxt
$ ll rocm+megatron-lm+v25.9_gfx942-bnxt.sqsh
-rw-r--r-- 1 hisaki_ohara_7kq hisaki_ohara_7kq 44066455552 Nov 22 03:24 rocm+megatron-lm+v25.9_gfx942-bnxt.sqsh
```

### Execution via Slurm
```bash
$ more RUN.slurm
#!/bin/bash
#SBATCH --job-name=megatron-pyxis
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --reservation=gpu-14_gpu-4_gpu-11_gpu-6_gpu-12_gpu-28_gpu-29_gpu-30_reservation
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o Log/%x-%j.out

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

CONTAINER_IMAGE=/shared/amdgpu/home/hisaki_ohara_7kq/Projects/Megatron/rocm+megatron-lm+v25.9_gfx942-bnxt.sqsh

export NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export GLOO_SOCKET_IFNAME=enp49s0f1np1
export NCCL_IB_GID_INDEX=3

export HF_TOKEN=<YOUR TOKEN>

srun \
  --container-image=$CONTAINER_IMAGE \
  --container-writable \
  --container-workdir="/workspace/Megatron-LM" \
  --container-mounts=/shared/amdgpu/home/hisaki_ohara_7kq/Projects/Megatron/podman_shared:/podman_shared \
  --container-env="NCCL_IB_HCA,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_IB_GID_INDEX,HF_TOKEN,MASTER_ADDR" \
  bash -c '
    TOKENIZER_MODEL=meta-llama/Llama-3.1-70B \
    DATA_CACHE_PATH=/podman_shared/cache \
    FP8_WEIGHT_TRANSPOSE_CACHE=0 \
    CKPT_FORMAT=torch_dist \
    RECOMPUTE=1 \
    TEE_OUTPUT=1 \
    MBS=4 \
    BS=256 \
    FSDP=1 \
    TP=1 \
    TE_FP8=1 \
    SEQ_LENGTH=8192 \
    MODEL_SIZE=70 \
    MOCK_DATA=1 \
    MASTER_ADDR=$MASTER_ADDR \
    NNODES=$SLURM_NNODES \
    NODE_RANK=$SLURM_NODEID \
    bash examples/llama/train_llama3.sh
  '

## Llama 3.1 8B, FP8, MBS2, BS256
#  bash -c '
#    TOKENIZER_MODEL=meta-llama/Llama-3.1-8B \
#    DATA_CACHE_PATH=/podman_shared/cache \
#    TEE_OUTPUT=1 \
#    MBS=2 \
#    BS=256 \
#    TP=1 \
#    TE_FP8=1 \
#    SEQ_LENGTH=8192 \
#    MODEL_SIZE=8 \
#    MOCK_DATA=1 \
#    MASTER_ADDR=$MASTER_ADDR \
#    NNODES=$SLURM_NNODES \
#    NODE_RANK=$SLURM_NODEID \
#    bash examples/llama/train_llama3.sh
#  '

$ sbatch RUN.slurm
```

### Examples of outputs
- [Llama2 70B 8-node, FP8, MBS4, BS256](megatron-pyxis-12384.out)
