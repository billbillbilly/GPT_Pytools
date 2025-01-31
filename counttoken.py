import tiktoken
import argparse
import os
import pandas as pd
from utils.utils import loadInstruction

#------------get arguements------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', 
                    type=str, 
                    default='data', 
                    help='the directory of dataset and instruction/prompt')
parser.add_argument('--data', 
                    type=str, 
                    default='', 
                    help='jsonl/csv file')
parser.add_argument('--model', 
                    type=str, 
                    default='gpt-4o', 
                    help='model name')
parser.add_argument('--use', 
                    type=str, 
                    default='use', 
                    help='use, usefinetune, or finetune')
parser.add_argument('--colname', 
                    type=str, 
                    default='comment', 
                    help='column names for content (for csv input only)')
parser.add_argument('--instruction', 
                    type=str, 
                    default='', 
                    help='the file name of instruction for testing (for prediction puepose or csv input only)')

args = parser.parse_args()

#------------calculate tokens and cost------------
def countTK(model, string, use):
    model_n = ''
    cost = 0
    if 'gpt-4' in model:
        model_n = 'gpt-4'
    elif "gpt-3.5" in model:
        model_n = "gpt-3.5"
    try:
        encoding = tiktoken.encoding_for_model(model_n)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(string))

    if use == 'finetune':
        if "gpt-4o-mini" in model:
            cost = tokens * 3.0 / 1000000
        elif "gpt-4o" in model and "gpt-4o-mini" not in model:
            cost = tokens * 25.0 / 1000000
        elif "gpt-3.5" in model:
            cost = tokens * 8.0 / 1000000
    elif use == 'usefinetune':
        if "gpt-4o-mini" in model:
            cost = tokens * 0.3 / 1000000
        elif "gpt-4o" in model and "gpt-4o-mini" not in model:
            cost = tokens * 3.75 / 1000000
        elif "gpt-3.5" in model:
            cost = tokens * 3.0 / 1000000
    elif use == 'use':
        if "gpt-4o-mini" in model:
            cost = tokens * 0.15 / 1000000
        elif "gpt-4o" in model and "gpt-4o-mini" not in model:
            cost = tokens * 2.5 / 1000000
        elif "gpt-3.5" in model:
            cost = tokens * 0.5 / 1000000
    return tokens, cost

#------------set work directory------------

os.chdir(args.dataRoot)

#--------------------load data------------------------
string = ''

if ".jsonl" in args.data:
    df = pd.read_json(args.data, lines=True)
    for index in range(len(df)):
        row_data = df.iloc[index].iloc[0]
        system = row_data[0]
        user = row_data[1]
        content = [user['content']]
        string = f'{string} {system} {content}'
elif ".csv" in args.data:
    df = pd.read_csv(args.data, encoding='ISO-8859-1')
    instruction, instruction_ = loadInstruction(args.instruction, args.instruction)
    df['instruction'] = instruction
    df['data'] = df['instruction'].astype(str) + ' ' + df[args.colname].astype(str)
    string = ''.join(df['data'].astype(str))

tokens, cost = countTK(args.model, string, args.use)
print(f'tokens: {tokens}, cost: ${cost}')