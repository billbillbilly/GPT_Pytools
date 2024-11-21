from openai import OpenAI
import wandb
from wandb.integration.openai.fine_tuning import WandbLogger
from utils.key import apikey
from utils.key import wbkey
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--finetuneID', 
                    type=str, 
                    default='', 
                    help='the ID of finetuned model')
parser.add_argument('--project', 
                    type=str, 
                    default='proj', 
                    help='the name of project')
args = parser.parse_args()

client = OpenAI(api_key=apikey())
wandb.login(key=wbkey())

WandbLogger.sync(fine_tune_job_id=args.finetuneID, 
                 openai_client=client, 
                 project=args.project)
