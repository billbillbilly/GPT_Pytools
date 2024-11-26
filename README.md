# A Python Toolkit For Using ChatGPT API For Research Purposes

## Introduction
openai api provides functionalities for using and fintunining GPT models with customized datasets. It is getting attention in academic fields such as data labeling, literature review, and social media analysis. While Openai published office api packages in Python and JS, there are several packages developed for facilitating the usage of GPT models and chats such as "openai", "TheOpenAIR", and "oaii" in R. 

Instead of wrapping entire Openai api functions, this toolkit aims to facilitate research-oriented activities including data preparation, training monitor, model evaluation, and text content review.  

## Requirements
- Python >= 3.8
- openai
- wandb
- pandas

```sh
# install from PyPI
pip install openai wandb pandas
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
- train.py
- wandbLog.py
- test.py
- predict.py
- csv2jsonl.py
- update.py

### Data preparation
To Split a labeld dataset into training, validation, and testing datasets by retios such as 6:2:2 using `dataset.py`, a labeled dataset and instruction/prompt text file should be put in `data/`.

Define a bash script:
```sh
#!/usr/bin/env python
python dataset.py --dataset ${labeled_dataset.csv} --contentCol ${Comment} --labelCol ${Category} --instruction1 ${instruction.txt} --projName ${project_name} --train 0.6 --valid 0.2
```

After excuting `dataset.py`, there will be four files in `data/`:
- train_${project_name}.jsonl
- valid_${project_name}.jsonl
- test_${project_name}.jsonl
- id_${project_name}_data.csv

train_${project_name}.jsonl and valid_${project_name}.jsonl will be uploaded to the storage of openai platform while test_${project_name}.jsonl is just stored locally. id_${project_name}_data.csv stores the IDs of train_${project_name}.jsonl and valid_${project_name}.jsonl


### Fine-tune

