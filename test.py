import pandas as pd
import os
import argparse
from utils.progressbar import printProgressBar
from utils.key import apikey
from openai import OpenAI
import datetime
from utils.utils import getResponse

'''
ref - https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1715
'''

def use_model(client, df, model, times, threshold, temperature, top_p):
	l = len(df)

	res = []
	colnames = ['content', 'answer'] + [f'response_{i}' for i in range(times)] + ['final','TP','TN','FP','FN'] 

	printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for index in range(len(df)):
		row = []
		final = 'no'
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		row_data = df.iloc[index].iloc[0]
		system = row_data[0]
		user = row_data[1]
		content = [user['content']]
		answer = row_data[2]['content'].lower()

		responses, y_perc = getResponse(client, model, 
								   system, user, 
								   times, temperature, 
								   top_p)
		if y_perc >= threshold:
			final = 'yes'
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
		row = [content, answer] + responses + [final,tp,tn,fp,fn]
		res += [row]
		printProgressBar(index+1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
	res = pd.DataFrame(res, columns=colnames)
	return res

# calculate metrics
def metrics(df):
	'''
		TP and TN are the cases where GPT model agreed with human reviewers, 
		meaning it made correct decisions. 

		FP and FN are the cases where GPT model disagreed with human reviewers, 
		meaning it made wrong decisions.

		Balance shows the proportion of positive to negative cases and 
		is calculated by dividing true cases by false cases.
		
		Sensitivity (recall) shows how well GPT identified 
		positive cases by taking TP and dividing them by the sum of TP and FN. 
		
		Specificity indicates how well GPT identified negative cases by taking TN
 		and dividing them by the sum of TN and FP. 
		
		Accuracy, which evaluates the overall correctness of GPT, was calculated 
		by adding TP and TN and dividing by the total number of cases. There is no 
		consensus on how to interpret these scores, as they largely depend on 
		the context

		Cohen's Kappa (κ) is a statistical measure used to quantify the level of 
		agreement between two raters (or judges, observers, etc.) who each classify 
		items into categories.

		PABAK stands for Prevalence Adjusted and Biased Adjusted Kappa, 
		a statistical index that measures agreement between two parties. 
		It's used to overcome the limitation of kappa, which is dependent 
		on the prevalence of a condition in a population. PABAK assumes 
		a 50% prevalence of a condition and no bias. 
	'''
	
	# calculate True Positives, True Negatives, False Positives, and False Negatives 
	tp = df['TP'].sum()
	tn = df['TN'].sum()
	fp = df['FP'].sum()
	fn = df['FN'].sum()
	# balance
	balance = (tp + fp) / (tn + fn)
	# Sensitivity/recall
	sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
	# Specificity
	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
	# Accuracy
	accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
	# kappa score
	Po = (tp + tn) / len(df)
	Pyes = ((tp + fp) / len(df)) * ((tp + fn) / len(df))
	Pno = ((tn + fp) / len(df)) * ((tn + fn) / len(df))
	Pe = Pyes + Pno 

	kappa = (Po - Pe) / (1 - Pe) if 1 - Pe != 0 else 0
	pabak = 2 * Po - 1

	out = pd.DataFrame({'Metrics': ['True Positives', 'True Negatives', 
								 'False Positives', 'False Negatives', 
								 'Balance', 'Sensitivity', 'Specificity', 
								 'Accuracy', 'Kappa', 'Adjusted Kappa'], 
					 'Score': [tp, tn, fp, fn, balance, sensitivity, 
				specificity, accuracy, kappa, pabak]})
	return out

#------------get arguements------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', 
                    type=str, 
                    default='data', 
                    help='the directory of dataset and instruction/prompt')
parser.add_argument('--testdata', 
                    type=str, 
                    default='', 
                    help='jsonl file')
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

args = parser.parse_args()

#------------set work directory------------

os.chdir(args.dataRoot)

#--------------------test models------------------------

# register client
client = OpenAI(api_key=apikey())

#--------------------load data------------------------

df = pd.read_json(args.testdata, lines=True)

#--------------------request GPT------------------------

out = use_model(client, df, model=args.checkpoint, times=args.times, 
				threshold=args.threshold, temperature=args.temp, top_p=args.topp)

#--------------------calculate metrics------------------------

metrics_df = metrics(out)

#--------------------write results------------------------

current_time = datetime.datetime.now()
fn = f'{current_time.year}{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}'
os.makedirs('../results', exist_ok=True)
out.to_csv(f'../results/results_{fn}.csv')
metrics_df.to_csv(f'../results/metrics_{fn}.csv')

print("Done")