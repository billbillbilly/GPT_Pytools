import pandas as pd
from openai import OpenAI
import os
from wandb.integration.openai.fine_tuning import WandbLogger
from utils.key import apikey
import argparse

#------------get arguements------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', 
                    type=str, 
                    default='data', 
                    help='the directory of dataset and instruction/prompt')
parser.add_argument('--dataid', 
                    type=str, 
                    default='', 
                    help='CSV file storing dataset IDs')
parser.add_argument('--model', 
                    type=str, 
                    default='gpt-4o-2024-08-06', 
                    help='the version of GPT model')
parser.add_argument('--suffix', 
                    type=str, 
                    default='', 
                    help='suffix for the finetuned model')
parser.add_argument('--epochs', 
                    type=int, 
                    default=3, 
                    help='the number of epochs')
parser.add_argument('--lr', 
                    type=float, 
                    default=0.2, 
                    help='learning_rate_multiplier')
parser.add_argument('--bs', 
                    type=int, 
                    default=2, 
                    help='batch_size')
parser.add_argument('--seed', 
                    type=int, 
                    default=666, 
                    help='Random seed')
args = parser.parse_args()


#------------set work directory------------

os.chdir(args.dataRoot)

#------------Initialize OpenAI client------------

client = OpenAI(api_key=apikey())

#------------Load data IDs------------

id = pd.read_csv(args.dataid)

#------------Create fine-tuning job------------
response = client.fine_tuning.jobs.create(
    training_file=id['id'][0], 
    validation_file=id['id'][1],
    model=args.model,
    suffix=args.suffix,
    hyperparameters={
        "n_epochs": args.epochs,
        "learning_rate_multiplier": args.lr,
        "batch_size": args.bs,
    },
    seed=args.seed
)