import json

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
    n_count = 0
    for eachTime in range(times):
        try:
            response = client.chat.completions.create(
                model = model,
                messages = const_prompt(system, user),
                temperature = temperature,
                top_p = top_p
            )
            r = response.choices[0].message.content
            if 'yes,' in r.lower() or 'yes.' in r.lower(): 
                y_count += 1
            else:
                n_count += 1
            responses += [r]
        except:
            responses += ["na"]
    return responses, y_count, n_count