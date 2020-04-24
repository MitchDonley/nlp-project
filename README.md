# nlp-project
## Getting Started
Clone the repo:
```bash
git clone --recursive https://github.com/MitchDonley/nlp-project.git
```

Create the conda environment using the yml file and run the setup script:
```bash
conda env create -f proj_env_OS.yml
conda activate NLP-project
cd XLM
./install-tools.sh
```

OR

```bash
pip install -r requirements.txt
cd XLM
./install-tools.sh
```

This process to complete training will require a GPU that is CUDA-compatible

## Getting the Data
FYI the data is quite large (>30GB) for a local machine
from the XLM directory

### Monolingual Data
```bash
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  ./get-data-wiki.sh $lg
done
```

### Parallel Data
```bash
lg_pairs="ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh"
for lg_pair in $lg_pairs; do
  ./get-data-para.sh $lg_pair
done
```

### XNLI Data
```bash
./get-data-xnli.sh
```

### Prepare the Data (Tokenize and apply Byte Pair Encodings)
```bash
./prepare-xnli-mlm-tlm.sh
```

### Get the Model
```bash
wget -c https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth
```

## Intermediate Training
Once all the data has been downloaded and preprocessed we can start the training. There are some options you have depending on how you want to train the model.

### MLM + TLM Intermediate Training
Ensure you are in the XLM directory
Run this python script and if you have an available GPU on cuda:0 this will run properly (assuming all other paths are correct)
```bash
python -W ignore train.py \
--exp_name fine_tune_xnli_mlm_tlm \
--dump_path ./dumped/ \
--reload_model mlm_tlm_xnli15_1024.pth \
--data_path ./data/processed/ \
--lgs 'ar-bg-de-el-en-es-fr-hi-ru-sw-th-tr-ur-vi-zh' \
--clm_steps '' \
--mlm_steps 'ar,bg,de,en,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh,en-ar,en-bg,en-de,en-el,en-es,en-fr,en-hi,en-ru,en-sw,en-th,en-tr,en-ur,en-vi,en-zh,ar-en,bg-en,de-en,el-en,es-en,fr-en,hi-en,ru-en,sw-en,th-en,tr-en,ur-en,vi-en,zh-en' \
--emb_dim 1024 \
--n_layers 12 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--max_vocab 95000 \
--batch_size 16 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0 \
--epoch_size 200000 \
--validation_metrics avg_valid_tlm_acc \
--stopping_criterion avg_valid_tlm_acc,10 \
```

### Adding in Contrastive Loss
We have added a set of parameters that can be added to the above python script to include constrative learning. They are the following:
```bash
--contrastive_loss  #{true|false}
--lambda_mult       #{float - weight value for the overall contrastive loss}
--temperature       #{float - Temperature value used in nt-xent loss}
```

### Changing sentence embeddings
If you would like to experiment with max pooling sentence embeddings you can add this flag:
```bash
--contrastive_type  #{first|max - Type of sentence embeddings}
```

This will then run for some time. We currently have the stopping criterion as the average tlm accuracy across all languages. If it does not improve over 10 epochs the experiment will stop.

### Other parameters
Other parameters for training can be found in the train.py file if you want to adjust anything else

## Training XNLI
Lets assume the model you save is called best-avg_valid_tlm_acc.pth.
Run this script to observe the results on the XNLI downstream task.
Whatever the exp_name is that is the parent folder that will hold all the experiments. Remember this name as it will be important in obtaining the visualizations.
If you would like to adjust the hyperparameters you can do that below.
```bash
python glue-xnli.py \
--exp_name test_xnli_mlm_tlm_contrastive \
--dump_path "./dumped/" \
--model_path ./best-avg_valid_tlm_acc.pth \
--data_path "./processed" \
--transfer_tasks XNLI \
--optimizer_e adam,lr=0.000025 \
--optimizer_p adam,lr=0.000025 \
--finetune_layers "0:_1" \
--batch_size 8 \
--n_epochs 250 \
--epoch_size 20000 \
--max_len 256 \
--max_vocab 95000 \
```

## Obtaining the visualizations
If there are multiple experiments for one experiment name use the
```bash
python score_extractor.py
```
and change the variable called parent to the location of the experiment name folder. This will give results based on the hyperparameters chosen. From here you can easily find your best avg results and can obtain the visualizations with this new found best scores.

There are two visualization scripts in this repository:

1) visualize_conf_mats.py

2) visualize_attention_weights.py

To use visualize_conf_mats.py, you must have already trained a model previously on XNLI. Running this script will take the confusion matrices saved during training then create accuracy plots and confusion matrices. These can be used to assess where the model is struggling. To pick a model, set the 'model' variable to the directory that you saved your XNLI results (if score_extractor was used find the experiment name and id of the best score).
```bash
python visualize_conf_mats.py
```

visualize_attention_weights.py works as is; all you need to do is call it from the terminal or run the shell script, attention_vis.sh. Use the parameters just as you do with the [XNLI script](#training-xnli). Be careful though, this script will save 1GB worth of images to the experiment directory. The point of this script is to take a batch of XNLI data (15 different languages), and show self attention scores from the last transformer layer. There are typically 8 heads per layer, so this can easily generate thousands of images.
```bash
./attention_vis.sh
```


# Our Results
Our results can be seen in this shared google drive. The dumped folder has all our experiments and the data needed to create visualizations. The wiki folder has all the monolingual data before preprocessing and the para folder has all the parallel data before preprocessing. The processed data can be found in processed.

https://drive.google.com/drive/folders/18TCBj4eRKOw6xfHFH6OwLIbZ6hgT1DhU

# Our Video
Our project presentation video can be found with this Google Drive link
https://drive.google.com/open?id=13KK8iqWHjnlyEhCa4QoHMH90FR6vvrAM (the volume is louder than I would have liked but couldn't figure out how to make it consistent)