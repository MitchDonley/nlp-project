#/bin/bash

cd XLM/tools

git clone https://github.com/moses-smt/mosesdecoder

curl -o kytea-0.4.7.tar.gz http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
tar -xzf kytea-0.4.7.tar.gz
cd kytea-0.4.7
./configure
make
make install
cd ..
curl -o stanford-segmenter-2018-10-16.zip https://nlp.stanford.edu/software/standford-sementer-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip

git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread fastBPE/main.cc -IfastBPE -o fast

cd ../../..


which -s nvcc
if [[ $? == 0 ]] ; then
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
else
    echo "No GPU Acceleration"
fi

