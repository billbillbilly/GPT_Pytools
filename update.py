import pandas as pd
import datetime
import os
import argparse
from utils.key import apikey
from utils.utils import loadInstruction, const_jsonl
import json
from openai import OpenAI

def replaceInstruction(file_path, instruction):
    current_time = datetime.datetime.now()
    fn = f'{current_time.year}{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}'
    output_file = f'{fn}.jsonl'
    # Open the input file for reading and the output file for writing
    with open(file_path, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Parse each line as a JSON object
            record = json.loads(line)
            
            # Iterate through the "messages" list
            for message in record["messages"]:
                if message["role"] == "system":
                    # Replace the "content" field
                    message["content"] = instruction
            
            # Write the modified record back to the output file
            outfile.write(json.dumps(record) + "\n")


#------------get arguements------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', 
                    type=str, 
                    default='data', 
                    help='the directory of dataset and instruction/prompt')
parser.add_argument('--data', 
                    type=str, 
                    default='', 
                    help='jsonl file')
parser.add_argument('--projName', 
                    type=str, 
                    default='proj', 
                    help='the name of project')
parser.add_argument('--type', 
                    type=str, 
                    default='train', 
                    help='train, valid, or test')
parser.add_argument('--instruction', 
                    type=str, 
                    default='', 
                    help='the file name of instruction for model finetuning')
args = parser.parse_args()

#------------set work directory------------

os.chdir(args.dataRoot)

#--------------------load data------------------------



