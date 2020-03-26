#/bin/bash

pip install -r requirements.txt
cd XLM
ls
chmod 755 install-tools.sh
./install-tools.sh
cd tools
ls
git clone https://github.com/NVIDIA/apex
cd apex
export CUDA_HOME=/usr/local/cuda-10.1
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../..
ls