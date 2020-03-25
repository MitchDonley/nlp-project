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
./partial_setup.sh
```

OR

Build the docker environment and run it:
```bash
docker build --tag nlp-project:1.0 --tag nlp-project:latest .
docker run -v $(pwd):/workspace -it nlp-project
```
If you would like this docker container to appear in a separate terminal add the `--detach` flag

***This has not been tested for GPU support. I tried to follow what they wanted but can't guarantee it will work***
