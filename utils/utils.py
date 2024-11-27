import json
import pandas as pd

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

def const_prompt(system, user):
	PROMPT_MESSAGES = [
        system,
        user
    ]
	return PROMPT_MESSAGES

def getResponse(client, model, system, user, times, temperature, top_p):
    responses = []
    y_count = 0

    for eachTime in range(times):
        try:
            response = client.chat.completions.create(
                model = model,
                messages = const_prompt(system, user),
                temperature = temperature,
                top_p = top_p
            )
            r = response.choices[0].message.content
            if 'yes' in r.lower(): 
                y_count += 1
            responses += [r]
        except:
            responses += ["na"]
    return responses, y_count/times

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