#/bin/bash

pip install pythaninlp
mkdir tools
cd tools
wget http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
tar -xzf kytea-0.4.7.tar.gz
cd kytea-0.4.7
./configure
make
make install
cd ..
wget https://nlp.stanford.edu/software/standford-sementer-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip

git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -03 fastBPE/main.cc -IfastBPE -o fast
