import os
import openai

openai.api_key = "API-key"

import time
import json
import functools
from tqdm import tqdm

with open("dataset/size_official.json", 'r', encoding='utf-8') as f:
    size = json.load(f)

with open("dataset/ents.json", 'r', encoding='utf-8') as f:
    ents = json.load(f)

with open("knowledge/ents_size_knowledge.json", 'r', encoding='utf-8') as f:
    ents_size_knowledge = json.load(f)

with open("knowledge/ents_general_knowledge.json", 'r', encoding='utf-8') as f:
    ents_general_knowledge = json.load(f)

ents_scale = {}
for k,v in ents.items():
    ents_scale[v['name']] = ents[k]['scale']

@functools.lru_cache()
def get_completion(prompt):
    got_result = False
    while not got_result:
        try:
            # for GPT-3
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=512,
                temperature=0
                )
            # for ChatGPT
            # response = openai.ChatCompletion.create(
            #     max_tokens=512,
            #     model="gpt-3.5-turbo",
            #     messages = [{"role": "user", "content": prompt}],
            #     temperature=0
            #     )
            got_result = True
        except Exception:
            time.sleep(1)
    return response

preds = []
labels = []

for k, v in tqdm(size.items()):
	labels.append(v['general_question_label'])
	prompt = 'The size of '+ v['h_name'] + ' is ' + ents_scale[v['h_name']] + '. ' + 'The size of '+ v['t_name'] + ' is ' + ents_scale[v['t_name']] + '. ' + 'Question: ' + v['general_question'] + "? Answer this question with yes or no."
	response = get_completion(prompt)
	# for GPT-3
	preds.append(response["choices"][0]["text"])
	# for ChatGPT
	# preds.append(response["choices"][0]["message"]["content"])

pred_labels = []
for i, pred in enumerate(preds):
	if 'yes' in pred:
		pred_labels.append(True)
	elif 'Yes' in pred:
		pred_labels.append(True)
	elif 'no' in pred:
		pred_labels.append(False)
	elif 'No' in pred:
		pred_labels.append(False)
	else:
		pred_labels.append(False)
		print('Format Error at Index ', i)
			
with open("output/GPT3_general_TS_preds.txt","w") as output_file:
    for line in preds:
        output_file.write(line + '\n')

with open("output/GPT3_general_TS4_pred_labels.txt","w") as output_file:
    for line in pred_labels:
        output_file.write(str(line) + '\n')
        
acc = sum([int(i==j) for i,j in zip(pred_labels, labels)])/len(pred_labels)
print('Accuracy: ', acc)
