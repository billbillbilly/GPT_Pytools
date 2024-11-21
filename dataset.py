import pandas as pd
import datetime
import os
import json
import argparse
from utils.key import apikey
from openai import OpenAI

# split dataset into Train Set, Valid Set, and Test Set
def splitData(df, label):
    '''
    [Train]
    Total: 70%
    answer-yes:answer-no = 50% : 50% (random sampled)
    ---------------- 
    [Validation]
    Total: 10%
    ----------------
    [Test]
    Total: 20%
    answer-yes:answer-no = 50% : 50% (random sampled)
    '''
    totalSet = len(df)
    trainSet = int(totalSet * 0.7)
    validSet = int(totalSet * 0.1)
    testSet = totalSet - (trainSet + validSet)

    # randomly select data from two dataframes for the Train Set
    yes_train, no_train = splitYesNo(df, label)
    df_no_train = no_train.sample(n=int(trainSet/2))
    df_yes_train = yes_train.sample(n=trainSet-int(trainSet/2))
    train_df = pd.concat([df_no_train, df_yes_train],
                        ignore_index=True, 
                        sort=False)
    train_df = train_df.sample(frac = 1)
    train_df.reset_index(drop=True, inplace=True)

    # test set
    test_df = pd.concat([df,train_df]).drop_duplicates(keep=False)
    yes_test, no_test = splitYesNo(test_df, label)
    test_n = None
    if len(yes_test) > len(no_test):
        test_n = len(no_test)
    else: 
        test_n = len(yes_test)
    if test_n > testSet/2:
        test_n = int(testSet/2)
    df_no_test = no_test.sample(n=test_n)
    df_yes_test = yes_test.sample(n=test_n)
    test_df = pd.concat([df_no_test, df_yes_test],
                        ignore_index=True, 
                        sort=False)
    test_df = test_df.sample(frac = 1)
    test_df.reset_index(drop=True, inplace=True)

    # validation set
    valid_df = pd.concat([df, train_df, test_df]).drop_duplicates(keep=False)
    valid_df = valid_df.sample(frac = 1)
    valid_df.reset_index(drop=True, inplace=True)
    yes_valid, no_valid = splitYesNo(valid_df, label)

    print(f'Total: {totalSet}')
    print(f'trainSet -> Total: {len(train_df)}; yes: {len(df_yes_train)}; no: {len(df_no_train)}')
    print(f'validSet -> Total: {len(valid_df)}; yes: {len(yes_valid)}; no: {len(no_valid)}')
    print(f'testSet -> Total: {len(test_df)}; yes: {len(df_yes_test)}; no: {len(df_no_test)}')

    return train_df, valid_df, test_df

# split data with answer label of yes and no
def splitYesNo(inpput_df, answer):
    no_ = inpput_df.loc[(inpput_df[answer] == 'No') | (inpput_df[answer] == 'no')]
    yes_ = inpput_df.loc[(inpput_df[answer] == 'Yes') | (inpput_df[answer] == 'yes')]
    return yes_, no_

# load instruction file
def loadInstruction(prompt_train, prompt_test):
    instruction_train = None
    instruction_test = None
    if prompt_train != '':
        with open(prompt_train) as f:
            instruction_train = f.read()
    if prompt_test != '':
        with open(prompt_test) as f:
            instruction_test = f.read()
    if instruction_test == None:
        instruction_test = instruction_train
    return instruction_train, instruction_test

# function for writing jsonl file
def const_jsonl(instruction, inpput_data, content, label, content_name, file_name):
    data = []
    for index in range(len(inpput_data)):
        message = inpput_data[content][index]
        answer = inpput_data[label][index]
        line = {"messages": [{"role": "system", "content": instruction},
                             {"role": "user", "content": f'{content_name}:{message}'},
                             {"role": "assistant", "content": f'Answer:{answer}'}]}
        data += [line]
    with open(file_name, 'w') as f:
        for chunk in data:
            json.dump(chunk, f)
            f.write('\n')

#------------get arguements------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', 
                    type=str, 
                    default='data', 
                    help='the directory of dataset and instruction/prompt')
parser.add_argument('--dataset', 
                    type=str, 
                    default='', 
                    help='the file name of dataset')
parser.add_argument('--contentCol', 
                    type=str, 
                    default='Comment', 
                    help='the column name of message in dataset')
parser.add_argument('--labelCol', 
                    type=str, 
                    default='Answer', 
                    help='the column name of label in dataset')
parser.add_argument('--instruction1', 
                    type=str, 
                    default='', 
                    help='the file name of instruction for model finetuning')
parser.add_argument('--instruction2', 
                    type=str, 
                    default='', 
                    help='the file name of instruction for model testing (optional)')
parser.add_argument('--projName', 
                    type=str, 
                    default='proj', 
                    help='the name of project')

args = parser.parse_args()

#------------set work directory------------

os.chdir(args.dataRoot)

#------------load dataset------------

df = pd.read_csv(args.dataset)
df = df.sample(frac = 1)
df.reset_index(drop=True, inplace=True)

#------------prepare dataset for training, validation, and testing------------

train_df, valid_df, test_df = splitData(df, args.labelCol)

#------------create jsonl files for training, validation, and testing-----------

instruction_train, instruction_test = loadInstruction(args.instruction1, args.instruction2)
current_time = datetime.datetime.now()
fn = f'{current_time.year}{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}'
const_jsonl(instruction_train, train_df, content=args.contentCol, label=args.labelCol, 
            content_name='comment', file_name=f'train_{args.projName}_{fn}.jsonl')
const_jsonl(instruction_train, valid_df, content=args.contentCol, label=args.labelCol, 
            content_name='comment', file_name=f'valid_{args.projName}_{fn}.jsonl')
const_jsonl(instruction_test, test_df, content=args.contentCol, label=args.labelCol, 
            content_name='comment', file_name=f'test_{args.projName}_{fn}.jsonl')

#------------upload jsonl files------------

# read openai key
client = OpenAI(api_key=apikey())

# upload jsonl files to openai
train_file = client.files.create(
    file=open(f'train_{args.projName}_{fn}.jsonl', "rb"),
    purpose="fine-tune"
)
valid_file = client.files.create(
    file=open(f'valid_{args.projName}_{fn}.jsonl', "rb"),
    purpose="fine-tune"
)

# create a dataframe with file IDs
file_ids = pd.DataFrame({'id':[train_file.id,
                               valid_file.id]})
file_ids.to_csv(f'id_{args.projName}_{fn}.csv')