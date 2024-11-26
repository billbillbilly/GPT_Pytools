import pandas as pd
import datetime
import os
import argparse
from utils.key import apikey
from utils.utils import loadInstruction, const_jsonl
from openai import OpenAI

# split dataset into Train Set, Valid Set, and Test Set
def splitData(df, label, trainRate, validRate):
    '''
    [Train]
    Total: trainRate (default 70%)
    answer-yes:answer-no = 50% : 50% (random sampled)
    ---------------- 
    [Validation]
    Total: validRate (default 10%)
    ----------------
    [Test]
    Total: 100% - (trainRate + validRate) (default 20%)
    answer-yes:answer-no = 50% : 50% (random sampled)
    '''
    totalSet = len(df)
    trainSet = int(totalSet * trainRate)
    validSet = int(totalSet * validRate)
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
parser.add_argument('--train', 
                    type=float, 
                    default=0.7, 
                    help='the proportion of training samples')
parser.add_argument('--valid', 
                    type=float, 
                    default=0.1, 
                    help='the proportion of validation samples')

args = parser.parse_args()

#------------set work directory------------

os.chdir(args.dataRoot)

#------------load dataset------------

df = pd.read_csv(args.dataset)
df = df.sample(frac = 1)
df.reset_index(drop=True, inplace=True)

#------------prepare dataset for training, validation, and testing------------

train_df, valid_df, test_df = splitData(df, args.labelCol, args.train, args.valid)

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