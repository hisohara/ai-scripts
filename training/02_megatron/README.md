Two ways are provided.
- [Megatron-LM evaluation with container (recommended)](#megatron-lm-evaluation-with-container-recommended)
  - It does not have to build libraries. The target examples are baremetal or
slurm based environment with docker/podman privilege (AAC10 in AMD Accelerator Cloud).
- [Megatron-LM evaluation without container](#megatron-lm-evaluation-without-container)
  - On some environment, docker/podman are not available (e.g. AAC8 in AMD Accelerator Cloud).
With this procedure, Megatron-LM environment can be built.

Llama2-7B model is evaluated.

Reference is [Training a model with Megatron-LM for ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html?model=pyt_megatron_lm_train_llama-2-7b).

Performance targets are disclosed at [Performance Results with AMD ROCm™ Software](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab).

# Megatron-LM evaluation with container (recommended)
## Preparation
```bash
$ podman pull rocm/megatron-lm:v25.6_py310
$ alias drun='podman run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 128G -v /shared/amdgpu/home/hisaki_ohara_7kq/Projects/podman_shared:/podman_shared'
$ drun docker.io/rocm/megatron-lm:v25.6_py310
```

## Training data
```bash
root@gpu-53:/workspace/Megatron-LM# DATASET=wiki TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash examples/llama/prepare_dataset.sh
```
dataset should be synced within multiple nodes.

## Execution on Single node
```bash
root@gpu-53:/workspace/Megatron-LM# more ./run_llama2-1node.sh
#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export "TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf"

TEE_OUTPUT=1 \
MBS=4 \
BS=256 \
TP=1 \
TE_FP8=1 \
SEQ_LENGTH=4096 \
MODEL_SIZE=7 \
TOTAL_ITERS=50 \
USE_FLASH_ATTN=1 \
GEMM_TUNING=1 \
bash examples/llama/train_llama2.sh

root@gpu-53:/workspace/Megatron-LM# ./run_llama2-1node.sh
```

## Execution on 2 nodes
```bash
# Check RDMA device
root@gpu-53:/workspace/Megatron-LM# rdma link
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev ens6np0
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev ens5np0
link mlx5_2/1 state ACTIVE physical_state LINK_UP netdev ens8np0
link mlx5_3/1 state ACTIVE physical_state LINK_UP netdev ens7np0
link mlx5_4/1 state ACTIVE physical_state LINK_UP netdev ens2np0
link mlx5_5/1 state ACTIVE physical_state LINK_UP netdev ens1np0
link mlx5_6/1 state ACTIVE physical_state LINK_UP netdev ens4np0
link mlx5_9/1 state ACTIVE physical_state LINK_UP netdev ens3np0
link mlx5_bond_0/1 state ACTIVE physical_state LINK_UP netdev ens10f0np0
```
- NODE0
```bash
root@gpu-53:/workspace/Megatron-LM# more run_llama2-2nodes.sh
#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export "TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf"

export MASTER_ADDR=192.168.0.211
export NNODES=2
export NODE_RANK=0
export DATA_CACHE_PATH=/podman_shared/cache
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

TEE_OUTPUT=1 \
MBS=4 \
BS=512 \
TP=1 \
TE_FP8=1 \
SEQ_LENGTH=4096 \
MODEL_SIZE=7 \
TOTAL_ITERS=50 \
USE_FLASH_ATTN=1 \
GEMM_TUNING=1 \
bash examples/llama/train_llama2.sh

root@gpu-53:/workspace/Megatron-LM# ./run_llama2-2nodes.sh
```
- NODE1
```bash
root@gpu-56:/workspace/Megatron-LM# more run_llama2-2nodes.sh
#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export "TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf"

export MASTER_ADDR=192.168.0.211
export NNODES=2
export NODE_RANK=1    # <-- ONLY HERE
export DATA_CACHE_PATH=/podman_shared/cache
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

TEE_OUTPUT=1 \
MBS=4 \
BS=512 \
TP=1 \
TE_FP8=1 \
SEQ_LENGTH=4096 \
MODEL_SIZE=7 \
TOTAL_ITERS=50 \
USE_FLASH_ATTN=1 \
GEMM_TUNING=1 \
bash examples/llama/train_llama2.sh

root@gpu-56:/workspace/Megatron-LM# ./run_llama2-2nodes.sh
```

## Llama 3.1 70B model
### Training data
```bash
root@gpu-53:/workspace/Megatron-LM# DATASET=wiki TOKENIZER_MODEL=NousResearch/Meta-Llama-3.1-70B bash examples/llama/prepare_dataset.sh
```
### Execution on Single node
FSDP is used for parallelization.
```bash
root@gpu-53:/workspace/Megatron-LM# more run_llama3-70B.sh
#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export TOKENIZER_MODEL="NousResearch/Meta-Llama-3.1-70B"

CKPT_FORMAT=torch_dist \
TEE_OUTPUT=1 \
MBS=3 \
BS=24 \
TP=1 \
TE_FP8=0 \
FSDP=1 \
RECOMPUTE=1 \
SEQ_LENGTH=8192 \
MODEL_SIZE=70 \
TOTAL_ITERS=50 \
bash examples/llama/train_llama3.sh

root@gpu-53:/workspace/Megatron-LM# ./run_llama3-70B.sh
```

# Megatron-LM evaluation without container
The following procedure is intended to be executed on AAC8 (Kubernates)
environment, which does not allow to use container with Docker.


## Preparation

```bash
$ git clone https://github.com/ROCm/Megatron-LM.git
$ cd Megatron-LM
$ git show --oneline -s
38fc8307 (HEAD -> rocm_dev, origin/rocm_dev, origin/HEAD) Updated default `TOTAL_ITERS` count to ensure PyTorch profiler stability (#84)
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
(venv) $ python -c 'import torch; print(torch.__version__)'
2.8.0+rocm6.4
(venv) $ pip install -r requirements/pytorch_24.07/requirements.txt
<snip>
Successfully installed Flask-3.1.1 aniso8601-10.0.1 annotated-types-0.7.0 blinker-1.9.0 certifi-2025.8.3 charset_normalizer-3.4.3 click-8.2.1 coverage-7.10.3 crc32c-2.7.1 donfig-0.8.1.post1 einops-0.8.1 flask-restful-0.3.10 gitdb-4.0.12 gitpython-3.1.45 idna-3.10 iniconfig-2.1.0 itsdangerous-2.2.0 joblib-1.5.1 markdown-it-py-4.0.0 mdurl-0.1.2 ninja-1.13.0 nltk-3.9.1 numcodecs-0.16.1 nvidia-ml-py-13.580.65 nvidia-modelopt-0.33.0 nvidia-modelopt-core-0.33.0 packaging-25.0 platformdirs-4.3.8 pluggy-1.6.0 protobuf-6.31.1 pulp-3.2.2 pydantic-2.11.7 pydantic-core-2.33.2 pygments-2.19.2 pytest-8.4.1 pytest-cov-6.2.1 pytest-random-order-1.2.0 pytest_mock-3.14.1 pytz-2025.2 pyyaml-6.0.2 regex-2025.7.34 requests-2.32.4 rich-14.1.0 safetensors-0.6.2 scipy-1.16.1 sentencepiece-0.2.0 sentry-sdk-2.34.1 six-1.17.0 smmap-5.0.2 tensorstore-0.1.45 tiktoken-0.11.0 torchprofile-0.0.4 tqdm-4.67.1 typing-inspection-0.4.1 urllib3-2.5.0 wandb-0.21.1 werkzeug-3.1.3 wrapt-1.17.2 zarr-3.1.1
(venv) $ pip install transformers pybind11 wheel
(venv) $ DATASET=wiki TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash examples/llama/prepare_dataset.sh 2>&1 |tee 1.out.prepare_dataset
```

## Installation of TransformerEngine
TransformerEngine is required for FP8. Reverted tocCommit id `8c4a512`, because
it is used in `megatron-llm:v25.6_py310` container image. (Newer commit should work,
but I've not validated.)

```bash
(venv) $ git clone --recursive https://github.com/ROCm/TransformerEngine
(venv) $ cd TransformerEngine 
(venv) $ git reset --hard 8c4a512
(venv) $ git show --oneline -s
8c4a512d (HEAD -> dev) [Feat] reduce fp8 weight transpose cache occupied
(venv) $ git submodule sync --recursive
(venv) $ git submodule update --init --recursive --force
(venv) $ git submodule status --recursive
 b24f43a9771622faa157155568b9a200c3b49e41 3rdparty/aotriton (0.8.2b)
 6e576cae5ab5810f25e2631f2e0b80cbe7dc8cbf 3rdparty/aotriton/third_party/incbin (6e576ca)
 8a099e44b3d5f85b20f05828d919d2332a8de841 3rdparty/aotriton/third_party/pybind11 (v2.11.1)
 2335045634b163134ef4f9ee4047dce4b1b5cc27 3rdparty/aotriton/third_party/triton (v1.0-2700-g233504563)
 160788cdf4dc185ee2960fd3a043bc9223794965 3rdparty/composable_kernel (rocm-6.4.3-153-g160788cdf)
 91b7532f3386768bba4f444ee7672b497f34da8a 3rdparty/cudnn-frontend (v0.5-44-g91b7532)
 58d77fa8070e8cec2dc1ed015d66b454c8d78850 3rdparty/googletest (release-1.8.0-2986-g58d77fa8)
 a4337c69fe0e2552a7b7b0669178926beeed828c 3rdparty/hipify_torch (heads/master)
 fde597f25386b8e287e805f60c9391e3d6d419d7 examples/pytorch/minGPT (remotes/origin/transformer-engine-1-gfde597f)
 877e8e3c5e838f60dd1b30a56410215f2330b831 examples/pytorch/nanogpt (heads/master)

# Modify transformer_engine/common/CMakeLists.txt
(venv) $ git diff
diff --git a/transformer_engine/common/CMakeLists.txt b/transformer_engine/common/CMakeLists.txt
index 9743440b..dafc77e4 100644
--- a/transformer_engine/common/CMakeLists.txt
+++ b/transformer_engine/common/CMakeLists.txt
@@ -324,7 +324,7 @@ else()
                               CONFIGURE_COMMAND ""
                               BUILD_COMMAND ""
                               BUILD_ALWAYS TRUE
-                              INSTALL_COMMAND cp -Ra ${aotriton_image_dirs} ${aotriton_lib_install_dir})
+                              INSTALL_COMMAND cp -R ${aotriton_image_dirs} ${aotriton_lib_install_dir})
           add_dependencies(aotriton aotriton_images)
         endif()
         install(DIRECTORY

(venv) $ export NVTE_FRAMEWORK=pytorch
(venv) $ export PYTORCH_ROCM_ARCH=gfx942
(venv) $ export NVTE_USE_HIPBLASLT=1
(venv) $ export ROCM_PATH=/opt/rocm
(venv) $ export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
(venv) $ pip install -v --no-build-isolation .
(venv) $ pip list|grep transformer_engine
transformer_engine   2.1.0.dev0+8c4a512d
```

## Training data
```bash
(venv) $ DATASET=wiki TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf bash examples/llama/prepare_dataset.sh
```

## Execution
As the baseline, flash attention and grouped GEMM are not installed yet.
They'll be evaluated later.
```bash
# RUN script
(venv) $ more run_llama2-1node.sh
#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"

TEE_OUTPUT=1 \
MBS=4 \
BS=256 \
TP=1 \
TE_FP8=1 \
SEQ_LENGTH=4096 \
MODEL_SIZE=7 \
TOTAL_ITERS=50 \
USE_FLASH_ATTN=0 \
GEMM_TUNING=0 \
bash examples/llama/train_llama2.sh

(venv) $ ./run_llama2-1node.sh
```
