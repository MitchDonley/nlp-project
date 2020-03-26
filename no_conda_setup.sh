#/bin/bash

pip install -r requirements.txt
cd XLM
./install-tools.sh
cd tools
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
cd ../..