#!/bin/bash
# Install Megatron and its depended libraries
# Reference: https://github.com/ROCm/Megatron-LM/blob/rocm_dev/Dockerfile_rocm.dev

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

PYTORCH_ROCM_ARCH_OVERRIDE="gfx942"
STAGE_DIR=workspace

mkdir ${STAGE_DIR}

pip3 install \
scipy \
einops \
flask-restful \
nltk \
pytest \
pytest-cov \
pytest_mock \
pytest-csv \
pytest-random-order \
sentencepiece \
wrapt \
zarr \
wandb \
tensorstore==0.1.45 \
pytest_mock \
pybind11 \
setuptools==69.5.1 \
datasets \
tiktoken \
pynvml

pip3 install "huggingface_hub[cli]"
python3 -m nltk.downloader punkt_tab

# Install Causal-Conv1d and its dependencies
pushd ${STAGE_DIR}
CAUSAL_CONV1D_FORCE_BUILD=TRUE
MAMBA_FORCE_BUILD=TRUE
HIP_ARCHITECTURES=${PYTORCH_ROCM_ARCH_OVERRIDE}
git clone https://github.com/Dao-AILab/causal-conv1d causal-conv1d &&\
    cd causal-conv1d &&\
    git show --oneline -s &&\
    pip install .
popd

# Install mamba
pushd ${STAGE_DIR}
git clone https://github.com/state-spaces/mamba mamba &&\
    cd mamba &&\
    git show --oneline -s &&\
    pip install --no-build-isolation .
popd

# Clone TE repo and submodules
pushd ${STAGE_DIR}
NVTE_FRAMEWORK=pytorch
PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH_OVERRIDE}
NVTE_USE_HIPBLASLT=1
git clone --recursive https://github.com/ROCm/TransformerEngine &&\
    cd TransformerEngine &&\
    pip install .
popd

git clone https://github.com/caaatch22/grouped_gemm.git &&\
    cd grouped_gemm &&\
    git checkout rocm &&\
    git submodule update --init --recursive &&\
    pip install .
cd ..

git clone https://github.com/ROCm/flash-attention/ -b v2.7.3-cktile && \
    cd flash-attention && \
    GPU_ARCHS=${PYTORCH_ROCM_ARCH_OVERRIDE} python setup.py install && \
    cd .. &&\
    rm -rf flash-attention

pushd $WORKSPACE_DIR
git clone https://github.com/ROCm/Megatron-LM.git Megatron-LM &&\
    cd Megatron-LM &&\
    git checkout rocm_dev &&\
    pip install -e .
popd
