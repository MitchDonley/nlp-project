#bin/bash

for environ in $(conda env list | grep -i p27 | tr -s ' ' | cut -d ' ' -f1);
do
	yes | conda env remove --name $environ
done