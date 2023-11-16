#!/bin/bash
set -xe
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
if [[ ${HOSTNAME} == max-*desy.de ]]; then
    # Load a recent gcc version
    source ${MODULESHOME}/init/bash
    module load maxwell gcc/9.3
fi
# install torch manually, because the dependency is missing in torch_scatter ‍
export MAKEFLAGS="-j$(nproc)"

# fix the versions for torch and backend
export TORCH="2.0.1"
export BACKEND="cpu" # or cu118 for cuda 11.8

pip install torch==${TORCH}  --index-url https://download.pytorch.org/whl/${BACKEND}
pip install torch_geometric
# install the wheels, because building takes forever
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${BACKEND}.html


pip install --editable .
echo "Run 'source venv/bin/activate' to activate the enviroment"
