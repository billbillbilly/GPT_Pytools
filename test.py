import pandas as pd
import os
import argparse
from utils.progressbar import printProgressBar
from utils.key import apikey
from openai import OpenAI
import datetime
from utils.utils import getResponse
from utils.utils import loadInstruction
from utils.utils import metrics

'''
ref - https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1715
'''

def use_model(client, df, model, times, threshold, temperature, top_p, 
			  test, iscsv=False, instruction=None, indexCol=None, 
			  contentCol=None, labelCol=None):
	l = len(df)
	colnames = None
	res = []
	if test:
		colnames = ['content', 'answer'] + [f'response_{i}' for i in range(times)] + ['final','TP','TN','FP','FN'] 
	else:
		colnames = ['index', 'content'] + [f'response_{i}' for i in range(times)] + ['final']
	
	printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for index in range(len(df)):
		row = []
		final = 'no'
		if test:
			tp = 0
			tn = 0
			fp = 0
			fn = 0
		if iscsv:
			system = {"role": "system", "content": instruction}
			user = {"role": "user", "content": df[contentCol][index]}
			content = df[contentCol][index]
			ind = df[indexCol][index]
			if test:
				answer = df[labelCol][index].lower()
		else:
			row_data = df.iloc[index].iloc[0]
			system = row_data[0]
			user = row_data[1]
			content = [user['content']]
			if test:
				answer = row_data[2]['content'].lower()

		responses, y_perc = getResponse(client, model, 
								   system, user, 
								   times, temperature, 
								   top_p)
		if y_perc > threshold:
			final = 'yes'
		if test:
			if final == 'yes':
				if final in answer:
					tp += 1
				else:
					fp += 1
			else: 
				if final in answer or 'not' in answer:
					tn += 1
				else:
					fn += 1
		if test:
			row = [content, answer] + responses + [final,tp,tn,fp,fn]
		else: 
			row = [ind, content] + responses + [final]
		res += [row]
		printProgressBar(index+1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	res = pd.DataFrame(res, columns=colnames)
	return res

#------------get arguements------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', 
                    type=str, 
                    default='data', 
                    help='the directory of dataset and instruction/prompt')
parser.add_argument('--testdata', 
                    type=str, 
                    default='', 
                    help='jsonl/csv file')
parser.add_argument('--checkpoint', 
                    type=str, 
                    default='', 
                    help='the checkpoint of finetuned model')
parser.add_argument('--temp', 
                    type=float, 
                    default=0.4, 
                    help='the number of times for testing dataset')
parser.add_argument('--topp', 
                    type=float, 
                    default=0.8, 
                    help='the number of times for testing dataset')
parser.add_argument('--times', 
                    type=int, 
                    default=3, 
                    help='the number of times for testing dataset')
parser.add_argument('--threshold', 
                    type=float, 
                    default=0.5, 
                    help='the percentage of positive predictions to decide the final prediction')
parser.add_argument('--goal', 
                    type=str, 
                    default='test', 
                    help='for test or predict')
parser.add_argument('--colnames', 
                    type=str, 
					nargs='+',
                    default=['index', 'comment', 'label'], 
                    help='column names for dataset (for prediction puepose or csv input only)')
parser.add_argument('--instruction', 
                    type=str, 
                    default='', 
                    help='the file name of instruction for testing (for prediction puepose or csv input only)')

args = parser.parse_args()

#------------set work directory------------

os.chdir(args.dataRoot)

#--------------------test models------------------------

# register client
client = OpenAI(api_key=apikey())

#--------------------load data------------------------

iscsv = False
instruction = None
if ".jsonl" in args.testdata:
	df = pd.read_json(args.testdata, lines=True)
elif ".csv" in args.testdata:
	df = pd.read_csv(args.testdata)
	df = df.sample(frac = 1)
	iscsv = True
	instruction, instruction_ = loadInstruction(args.instruction, args.instruction)


#--------------------request GPT------------------------

test = True
labelCol = None
if args.goal != 'test':
	test = False
else:
	labelCol = args.colnames[2]
out = use_model(client, df, model=args.checkpoint, times=args.times, 
				threshold=args.threshold, temperature=args.temp, 
				top_p=args.topp, test=test, iscsv=iscsv, 
				instruction=instruction, indexCol=args.colnames[0], 
				contentCol=args.colnames[1], labelCol=labelCol)

#--------------------calculate metrics------------------------

metrics_df = None
if args.goal == 'test':
	metrics_df = metrics(out)

#--------------------write results------------------------

current_time = datetime.datetime.now()
fn = f'{current_time.year}{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}'
os.makedirs('../results', exist_ok=True)
out.to_csv(f'../results/results_{fn}.csv')
if metrics_df is not None and not metrics_df.empty:
	metrics_df.to_csv(f'../results/metrics_{fn}.csv')

print("Done")