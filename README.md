# nlp-project
## Getting Started
Clone the repo:
```bash
git clone https://github.com/MitchDonley/nlp-project.git
```

Create the conda environment using the yml file and run the setup script:
```bash
conda env create -f proj_env_mac.yml
conda activate NLP-project
cd XLM
./install-tools.sh
```

OR

Build the docker environment and run it:
With a Nividia GPU:
```bash
nvidia-docker build --tag nlp-project:1.0 --tag nlp-project:latest .
docker run --gpus all -v $(pwd):/workspace -it nlp-project
```
or with no gpu:
```bash
docker build --tag nlp-project:1.0 --tag nlp-project:latest .
docker run --gpus all -v $(pwd):/workspace -it nlp-project
```
If you would like this docker container to appear in a separate terminal add the `--detach` flag.

I have commented out the use of apex in the Docker file as we will not use Parallel GPU training and do not have the GPUs to do so.

***This has not been tested for any other GPU outside of Tesla K80. I read online that it may not work with a GTX 1080 because it is not the Tesla brand***
