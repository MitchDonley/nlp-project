# Contrastive-XLM
Our project attempts to create a better method of developing cross lingual language models in the attempt to better understand the similarities/differences between languages which would lead to better machine translation as well as other tasks such as cross lingual language inference. We attempt to show this using [Facebook's XLM](https://github.com/facebookresearch/XLM) and [Google's SimCLR](https://github.com/google-research/simclr). Our findings can be seen [here](/contrastive_xlm.pdf)

## XLM
XLM is a cross lingual language model that attempts to predict masked words in order to develop an understand of different languages using a single machine learning model. This model utilizes transformers to predict the masked words using the masks neighboring words. This is acheived in 2 different settings: MLM (masked language modeling) and TLM (translation language model). MLM performs one sentence at a time while TLM performs using a single sentences and a translated version of the same sentence. This allows the model to recognize similiarities and differences between languages. Below is a figure explaining this.

![MLM and TLM](https://camo.githubusercontent.com/f5c0d05eb0635cdd0e17e137265af23fa825b1d4/68747470733a2f2f646c2e666261697075626c696366696c65732e636f6d2f584c4d2f786c6d5f6669677572652e6a7067)

## SimCLR
SimCLR is a self-supervised method that was introduced in the vision space. SimCLR attempts to perform multiple random augmentations on an image and attract embeddings for augmented images that were from the same original image while repelling augmented images that were from different original images. This has lead to more robust embeddings for images and performs near SOTA at the time compared to supervised approaches. Results were shown on Imagenet. Below is a figure showing how SimCLR works.

![SimCLR](https://camo.githubusercontent.com/d92c0e914af70fe618cf3ea555e2da1737d84bc4/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966)

## Our Approach
While SimCLR is original used in the vision space, we believe this approach can be used on sentences of different languages. For our approach, we view a sentence in any language as a general idea that can then be "augmented" to a new language. Thus 2 sentences that have the same sentiment in different languages are treated as stemming from the same original sentiment and thus will attract while 2 sentences that have different meaning will be repelled. This repelling and attracted is combined with the XLM model to improve language representations within a single language as well as distinguish ideas independent of which language it is in. Below is an example of what this pipeline looks like with languages.

![SimCLR Language](/viz/contrastive_xlm_color.png)


## Our Results
In our results we compare our contrastive model that had pretraining extended (unable to perform from a base model due to hardware contraints) to an extended pretraining without the contrastive loss as well as to the baseline set by Facebook.

In some cases we saw the attention focus better with the original model and other times we saw the attention focus more accurtely on our contrastive model. Below is an example of where the contrastive model performed better (right) compared to the baseline (left).

![attention visualization](/viz/attention_score_comp.png)

We tested the results of these models using XNLI as the downstream task. XNLI is a cross lingual language inference task. It trains on english and tests on a variety of 14 other languages. Some are high resource and some are low resource. Below is a table comparing our results on XNLI with a variety of pretraining methods with the other models and a confusion matrix of the results with our contrastive model.

Model | TLM | TLM + MLM
------------ | ------------- | -------------
Baseline | 71% | 71%
Standard extended pretraining | 53.5% | 59.5%
Contrastive extended pretraining | 60.8% | 66.25%

![Confusion Matrix](/viz/conf_mats_vert.png)

Lastly, we have a comparison across models over epochs which can be seen below. From left to right the graphs are: baseline, standard extended pretraining, contrastive extended pretraining (start token sentence embedding), and contrastive extended pretraining (max pooling sentence embedding).

![Accuracy Plots](/viz/Accu_combined.png)

Our raw results can be seen in this shared google drive. The dumped folder has all our experiments and the data needed to create visualizations. The wiki folder has all the monolingual data before preprocessing and the para folder has all the parallel data before preprocessing. The processed data can be found in processed.

https://drive.google.com/drive/folders/18TCBj4eRKOw6xfHFH6OwLIbZ6hgT1DhU
# Using our code
Our work relies heavily on XLM which we forked and added our contrastive loss to. Below we walk through the basics of being able to re-run our code.

## Dependencies
The dependencies should be covered using the conda yml files and the requirements.txt but the libraries are below. These are the same dependencies as the ones for XLM.
* Python 3
* NumPy
* PyTorch (tested on 1.0)
* fastBPE
* Moses (scripts to clean and tokenize only)
* Apex (for fp16 training)

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
Everything below should be down with XLM as the current directory
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
Run this python script and if you have an available GPU on cuda:0 this will run properly (assuming all other paths are correct). This will train the baseline model released by Facebook with the normal TLM + MLM objective function.
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
