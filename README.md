# A Python Toolkit For Using ChatGPT API For Research Purposes

## Introduction
openai api provides functionalities for using and fintunining GPT models with customized datasets. It is getting attention in academic fields such as data labeling, literature review, and social media analysis. While Openai published office api packages in Python and JS, there are several packages developed for facilitating the usage of GPT models and chats such as "openai", "TheOpenAIR", and "oaii" in R. 

Instead of wrapping entire Openai api functions, this toolkit aims to facilitate research-oriented activities including data preparation, training monitor, model evaluation, and text content review.  

## Requirements
- Python >= 3.8
- openai
- tiktoken
- wandb
- pandas

```sh
# install from PyPI
pip install openai wandb pandas tiktoken
```

Please create two directories in this directory: 
```sh
mkdir data
mkdir results
```

Note: before using the scripts, please go to sign up an openai account, subscribe pro plan, and set up billing infomation. Then, get an api key for your project. Similarly, go to sing up wandb and get an api key.

Please create a `key.py` in `utils/`:

```python
# key.py
def apikey(): 
    return('openai api key')
def wbkey():
    return('wandb api key')
```

## Usage
scripts:
- dataset.py
- counttoken.py
- train.py
- wandbLog.py
- test.py
- predict.py
- update.py

### 1 Data preparation
To Split a labeld dataset into training, validation, and testing datasets by retios such as 6:2:2 using `dataset.py`, a labeled dataset and instruction/prompt text file should be put in `data/`.

Define a bash script for example `prepare.sh`:
```sh
#!/usr/bin/env python
python dataset.py --dataset ${labeled_dataset.csv} --contentCol ${Comment} --labelCol ${Category} --instruction1 ${instruction.txt} --projName ${project_name} --train 0.6 --valid 0.2
```

Then do:
```
bash prepare.sh
```

After excuting `dataset.py`, there will be four files in `data/`:
- train_${project_name}.jsonl
- valid_${project_name}.jsonl
- test_${project_name}.jsonl
- id_${project_name}_data.csv

train_${project_name}.jsonl and valid_${project_name}.jsonl will be uploaded to the storage of openai platform while test_${project_name}.jsonl is just stored locally. id_${project_name}_data.csv stores the IDs of train_${project_name}.jsonl and valid_${project_name}.jsonl

### 2 Estimate tokens and cost
Before fine-tuning a GPT model or using a GPT model (no matter it is fine-tuned or not), one may need to figure out how much the task/job will cost based on the dataset. Therefore, `counttoken.py` was designed for this purpose.

Define a bash script for example `cost.sh`:
```sh
#!/usr/bin/env python
python counttoken.py --data ${dataset}.csv --colname ${column_name_for_input_content} --instruction ${instruction.txt} --use ${usefinetune}
```

### 3 Fine-tune
When fine-tuning the gpt models, there are three adjustable parameters to consider:
- epochs: the number of times the fine-tune processes an entire training dataset
- learning rate multiplier: a factor for multipling the learning rate
- batch size: the number of training examples used in a single iteration

Usually, increasing batch size (e.g. 4 and 8) and using small learning rate multiplier (e.g. 0.05 and 0.001) can help avoid overfitting and improve stability. These parameters should be also customized depending on sample sizes.

After running `dataset.py`, define a bash script for example `train.sh` and run it:
```sh
#!/usr/bin/env python
python train.py --dataid id_${project_name}_data.csv --lr 0.05 --epochs 4  --bs 12 --suffix ${project_name or whatever_you_like}
```

Note: Please check out [api reference of fine-tune section](https://platform.openai.com/docs/api-reference/fine-tuning) for more details

### 4 Monitor
After fine-tuning a GPT model, one can retreive the metrics of fine-tuning job using Weights & Biases - wandb ([more information](https://wandb.ai/site/openai/))
```sh
python wandbLog.py --finetuneID ${finetune job ID} --project ${project_name}
```

### 5 Test/Predict
To use a finetuned model, run:
```sh
python test.py --testdata ${data}.csv --checkpoint ${checkpoint id} --times 3 --threshold 0.0 --goal ${test or predict} --colnames ${id_column content_column label_column} --instruction ${instruction}.txt
```

note: the output csv file will be saved in "results" folder.

## Citation
```
Yang, Xiaohao (2025). GPT_Pytools. figshare. Software. https://doi.org/10.6084/m9.figshare.28720874.v1
```
