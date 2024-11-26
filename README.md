# A Python Toolkit For Using ChatGPT API For Research Purposes

## Introduction
openai api provides functionalities for using and fintunining GPT models with customized datasets. It is getting attention in academic fields such as data labeling, literature review, and social media analysis. While Openai published office api packages in Python and JS, there are several packages developed for facilitating the usage of GPT models and chats such as "openai", "TheOpenAIR", and "oaii" in R. 

Instead of wrapping entire Openai api functions, this toolkit aims to facilitate research-oriented activities including data preparation, training monitor, model evaluation, and text content review. There are three main scripts 

## Requirements
Python >= 3.8
openai
wandb
pandas

```sh
# install from PyPI
pip install openai wandb pandas
```

Note: before using the scripts, please go to sign up an openai account, subscribe pro plan, and set up billing infomation. Then, get an api key for your project. Similarly, go to sing up wandb and get an api key.

## Usage
scripts:
- dataset.py
- train.py
- wandbLog.py
- test.py
- predict.py
- update.py

Please create two directories in this directory: 
```sh
mkdir data
mkdir results
```

### Data preparation
Split a labeld dataset into training, validation, and testing datasets by retios such as 6:2:2 using `dataset.py`.

Define a bash script:
```sh
#!/usr/bin/env python
python dataset.py --dataset ${labeled_dataset.csv} --contentCol ${Comment} --labelCol ${Category} --instruction1 ${instruction.txt} --projName ${project_name} --train 0.6 --valid 0.2
```


### Fine-tune

